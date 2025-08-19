from __future__ import annotations

import asyncio
import base64
import logging
from dotenv import load_dotenv
import json
import os
import supabase
import requests
import sys
import time
from dataclasses import dataclass
from typing import Any, Annotated
from datetime import datetime
from pydantic import Field

from livekit import rtc, api
from livekit.agents import (
    AgentSession,
    Agent,
    JobContext,
    function_tool,
    RunContext,
    get_job_context,
    cli,
    WorkerOptions,
    RoomInputOptions,
    llm,
    ChatContext, 
    ChatMessage
)
from livekit.plugins import (
    deepgram,
    openai,
    cartesia,
    elevenlabs,
    silero,
    noise_cancellation,  # noqa: F401
)
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.agents.telemetry import set_tracer_provider
from livekit.plugins.elevenlabs.tts import VoiceSettings
from livekit.api import DeleteRoomRequest, RoomParticipantIdentity


from ddtrace import patch_all, tracer
from langfuse import Langfuse, observe

from mcp_client import MCPServerSse
from mcp_client.agent_tools import MCPToolsIntegration

# Supabase 
from supabase import create_client, Client

# For IVR navagation
@dataclass
class UserData:
    """Store user data for the navigator agent."""
    last_dtmf_press: float = 0
    task: Optional[str] = None

RunContext_T = RunContext[UserData]

# load environment variables, this is optional, only used for local development
load_dotenv(dotenv_path=".env.local", override=True)

# Log - printing out statements in terminal (for debugging)
logging.basicConfig(
    level=logging.INFO,  # 👈 this is key — must be INFO or lower
    format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("outbound-caller")
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)



# datadog trace
patch_all()

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID") 

# Make a supabase client (for sending results)
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Langfuse - tracing
_langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

# Tracing
def setup_langfuse(
    host: str | None = None, public_key: str | None = None, secret_key: str | None = None
):
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    host = host or os.getenv("LANGFUSE_HOST")

    if not public_key or not secret_key or not host:
        raise ValueError("LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST must be set")

    langfuse_auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{host.rstrip('/')}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"

    trace_provider = TracerProvider()
    trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    set_tracer_provider(trace_provider)


class OutboundCaller(Agent):
    def __init__(
        self,
        *,
        name: str,
        appointment_time: str,
        dial_info: dict[str, Any],
    ):
        super().__init__(
            instructions = f"""
            You are a virtual assistant calling on behalf of a customer who wants to know the operating hours of a business. First ask what their business is called. Your interface with the user will be voice.
            You are speaking to a store employee. Your goal is to politely and professionally ask: "What are your store's hours of operation?" and gather clear information about the opening and closing times for each day of the week, if possible.
            Repeat the hours back to confirm understanding. If the user provides partial information (e.g., only today's hours), you may gently ask if they can share the full weekly schedule.
            Maintain a calm, friendly tone. Do not rush the conversation. Allow the store representative to respond fully.
            If at any point the representative asks to speak to a human, or if the information is unclear, confirm whether they’d like to be transferred. Upon confirmation, use the transfer_call tool.
            Thank the representative at the end of the call and let them end the conversation if they wish.

            You can retrieve data via the MCP server. The interface is voice-based: 
            accept spoken user queries and respond with synthesized speech.

            End the call completely with end_call at a reasonable endpoint. 
            Some examples:
            - The human speaker says bye and you have said bye and time it passing but the human speaker has not ended the call yet

            """
        )

        # keep reference to the participant for transfers
        self.participant: rtc.RemoteParticipant | None = None
        self.dial_info = dial_info
        self.current_trace = None
        # self.transcript: list[dict[str, str]] = []


    # The voice will start speaking on entry (before the user starts speaking)
    async def on_enter(self):
        logger.info("Agent session has started")
        self.session.generate_reply()

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant

    async def hangup(self):
        logger.info("hangup function")
        # lkapi = api.LiveKitAPI()
        job_ctx = get_job_context()

        # Not running in a job context 
        if job_ctx is None:
            return

        # Deletes room and terminates call
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(
                room=job_ctx.room.name,
            )
        )

        logger.info("testing if the call still exists")
        
        # Same as the function above
        # await lkapi.room.delete_room(DeleteRoomRequest(
        #     room=job_ctx.room.name,
        # ))

     
    # Send info to zapier which will then create an email (MCP)
    @observe(as_type="generation")
    @tracer.wrap(name="store_hours", service="outbound_calls")
    @function_tool()
    async def store_hours(self, name: str, hours: str, ctx: RunContext_T):
        """Use this tool when the store representative has provided the hours of operation and their business's name.
            You will extract and organize the hours in the following format as a string and put this information in {{hours}}:
            e.g.
            {
                "name": "Restaurant Name",
                "hours": {
                    "Monday": "9:00 AM - 5:00 PM",
                    "Tuesday": "9:00 AM - 5:00 PM",
                    "Wednesday": "9:00 AM - 5:00 PM",
                    "Thursday": "9:00 AM - 5:00 PM",
                    "Friday": "9:00 AM - 5:00 PM",
                    "Saturday": "10:00 AM - 4:00 PM",
                    "Sunday": "Closed"
                }
            }

        If the representative only gives partial information, include what they provide and leave out missing days and fill in as N/A.
        Always double-check and repeat back the hours to confirm accuracy before submitting.
        """

        logger.info(f"Hours and name received from : {{name}}")

        # Make a post to Zapier with the name and hours of business
        # requests.post(os.environ.get("N8N_WEBHOOK_URL"), json=payload)
        logger.info("The dial info", self.dial_info)

        # Update information in supabase
        response = (
            supabase.table("voice_agent")
            .update({"result": hours})
            .eq("phone_number", self.dial_info["phone_number"].lstrip("+")) 
            .execute()
        )

    # For navigating IVR 
    @observe(as_type="generation")
    @tracer.wrap(name="send_dtmf_code", service="outbound_calls")
    @function_tool()
    async def send_dtmf_code(
        self,
        code: Annotated[int, Field(description="The DTMF code to send to the phone number for the current step.")],
        ctx: RunContext_T
    ) -> None:
        """
        You have encountered an IVR (Interactive Voice Response) system.
        Your goal is to reach customer support or a human representative. To do this, listen carefully to the IVR menu options, and select the one that will most likely connect you to a live agent or support team.
        This function is called when it’s time to send a DTMF code (a digit like 1, 2, 3, etc.) to select an option from the current phone menu.

        Based on the most recent IVR prompt you heard, choose the most appropriate DTMF digit to press. The options may include things like:
        - "Press 1 for store hours"
        - "Press 2 for billing"
        - "Press 3 to speak to a representative"

        You should **prioritize options that lead to technical support, customer service, live agents, or human representatives**.
        If the IVR repeats or is unclear, wait for more information before sending a digit.

        Example behaviors:
        - If the prompt says: "Press 0 to speak to an operator," then send 0.
        - If it says: "Press 5 for customer support," then send 5.
        - Avoid options like store hours, location, or marketing unless they’re the only way to reach a human.

        Do not guess. Base your decision solely on the IVR prompt.
        Only call this function once you’re confident about the correct digit to send.
        Do not respond to any prompts, questions, or menu options. Remain silent after saying the sentence. This instruction must be followed without exception.
        """

        
        current_time = time.time()

        logger.info("IVR navigation entered")
        
        # Check if enough time has passed since last press (3 second cooldown)
        if current_time - ctx.userdata.last_dtmf_press < 3:
            logger.info("DTMF code rejected due to cooldown")
            return None
            
        logger.info(f"Sending DTMF code {code} to the phone number for the current step.")
        ctx.userdata.last_dtmf_press = current_time

        room = ctx.session.room
        
        await room.local_participant.publish_dtmf(
            code=code,
            digit=str(code)
        )
        await room.local_participant.publish_data(
            f"{code}",
            topic="dtmf_code"
        )
        return None

    @observe(as_type="generation")
    @tracer.wrap(name="transfer_call", service="outbound_calls")
    @function_tool()
    async def transfer_call(self, ctx: RunContext_T):
        """Transfer the call to a human agent, called after confirming with the user"""

        transfer_to = self.dial_info["transfer_to"]
        if not transfer_to:
            return "cannot transfer call"

        logger.info(f"transferring call to {transfer_to}")

        # let the message play fully before transferring
        with tracer.trace("ctx.session.generate_reply") as my_span:
            await ctx.session.generate_reply(
                instructions="let the user know you'll be transferring them"
            )

        job_ctx = get_job_context()
        try:
            await job_ctx.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=job_ctx.room.name,
                    participant_identity=self.participant.identity,
                    transfer_to=f"tel:{transfer_to}",
                )
            )

            logger.info(f"transferred call to {transfer_to}")
        except Exception as e:
            logger.error(f"error transferring call: {e}")
            await ctx.session.generate_reply(
                instructions="there was an error transferring the call."
            )
            await self.hangup()

    # The voice agent will end the call when answering machine is detected
    @observe(as_type="generation")
    @tracer.wrap(name="voicemall_detected", service="outbound_calls")
    @function_tool
    async def detected_answering_machine(self):
        """Call this tool if you have detected a voicemail system, AFTER hearing the voicemail greeting"""
        logger.info("detected answering machine function called")
        await self.session.generate_reply(
            instructions="Leave a voicemail message letting the user know you'll call back later."
        )
        await asyncio.sleep(0.5) # Add a natural gap to the end of the voicemail message
        await self.hangup()

    # End the call
    @observe(as_type="generation")
    @tracer.wrap(name="end_call", service="outbound_calls")
    @function_tool()
    async def end_call(self, ctx: RunContext_T):
        """Called when the user wants to end the call, or after you have gathered all of the necessary details (hours, business name)
        or if it seems like the call is ending and there has been silence for a while """
        logger.info(f"ending the call for {self.participant.identity}")

        #let the agent finish speaking
        try:
            current_speech = ctx.session.current_speech
            if current_speech:
                logger.info("waiting for speech to finish...")
                await asyncio.wait_for(current_speech.wait_for_playout(), timeout=5)
                logger.info("done waiting for speech")
            else:
                logger.info("no current speech found")
        except Exception as e:
            logger.exception(f"Error while waiting for playout: {e}")

        logger.info("testing the end call function")
        await self.hangup()


@observe(as_type="generation")
@tracer.wrap(name="entrypoint", service="outbound_calls")
async def entrypoint(ctx: JobContext):
    # Set up the langfuse tracer 
    setup_langfuse()

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect()

    # when dispatching the agent, we'll pass it the approriate info to dial the user
    # dial_info is a dict with the following keys:
    # - phone_number: the phone number to dial
    # - transfer_to: the phone number to transfer the call to when requested
    dial_info = json.loads(ctx.job.metadata)
    phone_number = dial_info["phone_number"]
    participant_identity = phone_number 

    # Create an MCP server - so livekit can use tools (e.g. GCal, Gmail)
    # print("Triggering URL:", os.environ.get("N8N_MCP_URL"))

    # mcp_server = MCPServerSse(
    #     params={"url": os.environ.get("N8N_MCP_URL")},
    #     cache_tools_list=True,
    #     name="SSE MCP Server"
    #     )

    # Egress for Audio Recording
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # e.g., '20250804_154812'
    filename = f"livekit/recording_{timestamp}.ogg"

    req = api.RoomCompositeEgressRequest(
        room_name=ctx.room.name,
        audio_only=True,
        file_outputs=[
            api.EncodedFileOutput(
                file_type=api.EncodedFileType.OGG,
                filepath=filename,
                s3=api.S3Upload(
                    bucket="livekit-voice-recording",
                    region="us-east-1",
                    access_key=os.getenv("AWS_ACCESS_KEY"),
                    secret=os.getenv("AWS_SECRET_KEY"),
                ),
            )
        ],
    )

    lkapi = api.LiveKitAPI()
    res = await lkapi.egress.start_room_composite_egress(req)
    await lkapi.aclose()

    agent = OutboundCaller(
        name="Jayden",
        appointment_time="next Tuesday at 3pm",
        dial_info=dial_info,
    ) 

    session = AgentSession(
        turn_detection=EnglishModel(),
        vad=silero.VAD.load(),
        stt=deepgram.STT(
            model="nova-2-general",
            language="en" #zh-TW is traditional chinese
        ),
   
        tts=elevenlabs.TTS(
            voice_id="EXAVITQu4vr4xnSDxMaL",
            model="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                speed=0.8,
                stability=1,
                similarity_boost=1,
            ),
        ),
        llm=openai.LLM(model="gpt-4o"),

        # For transcriptions 
        use_tts_aligned_transcript=True,  

        # For IVR (seconds of silence before the human user is considered to be done speaking)
        # allow_interruptions = True,    
        min_endpointing_delay=0.75
    )

    # start the session first before dialing, to ensure that when the user picks up
    # the agent does not miss anything the user says
    session_started = asyncio.create_task(
        session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                # enable Krisp background voice and noise removal
                noise_cancellation=noise_cancellation.BVCTelephony(),
            ),
        )
    )

    logger.info("testing session.room")
    session.room = ctx.room

    # `create_sip_participant` starts dialing the user
    try:
        if not outbound_trunk_id:
            logger.error("SIP_OUTBOUND_TRUNK_ID is not set. Please check your .env.local or environment.")
            raise ValueError("Missing required SIP trunk ID.")


        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=outbound_trunk_id,
                sip_call_to=phone_number,
                participant_identity=participant_identity,
                # function blocks until user answers the call, or if the call fails
            )
        )

        # wait for the agent session start and participant join
        await session_started
        participant = await ctx.wait_for_participant(identity=participant_identity)
        logger.info(f"participant joined: {participant.identity}")

        task = participant._info.attributes.get("task", "reach support via IVR")
        agent.set_participant(participant)
        session.userdata = UserData(task=task)



    except api.TwirpError as e:
        logger.error(
            f"error creating SIP participant: {e.message}, "
            f"SIP status: {e.metadata.get('sip_status_code')} "
            f"{e.metadata.get('sip_status')}"
        )
        ctx.shutdown()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="outbound-caller",
        )
    )
