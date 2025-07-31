from __future__ import annotations

import asyncio
import base64
import logging
from dotenv import load_dotenv
import json
import os
import supabase
import requests
from typing import Any #Annotated
# from pydantic import Field
from datetime import datetime

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


from ddtrace import patch_all, tracer
from langfuse import Langfuse, observe
# from langfuse.decorators import observe

from mcp_client import MCPServerSse
from mcp_client.agent_tools import MCPToolsIntegration

# load environment variables, this is optional, only used for local development
load_dotenv(dotenv_path=".env.local", override=True)
logger = logging.getLogger("outbound-caller")
logger.setLevel(logging.INFO)

# Langfuse
_langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

# datadog trace
patch_all()

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")

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

        self.transcript: list[dict[str, str]] = []

        # Testing N8N webhook 
        # payload = {
        #     "restaurant_name": "Ten Thousand Coffee"
        # }

        # requests.post(os.environ.get("N8N_WEBHOOK_URL"), json=payload)



    # The voice will start speaking on entry (before the user starts speaking)
    async def on_enter(self):
        self.session.generate_reply()

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant

    async def hangup(self):
        """Helper function to hang up the call by deleting the room"""

        # Not running in a job context 
        if ctx is None:
            return

        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(
                room=job_ctx.room.name,
            )
        )

    # Send info to zapier which will then create an email (MCP)
    @observe(as_type="generation")
    @tracer.wrap(name="email_hours", service="outbound_calls")
    @function_tool()
    async def email_hours(self, name: str, hours: str, ctx: RunContext):
        """Use this tool when the store representative has provided the hours of operation and their business's name.
            You will extract and organize the hours in the following format:
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

        # Create payload for data being sent in post request 
        payload = {
            "name": name, 
            "hours:": hours
        }

        # Make a post to Zapier with the name and hours of business
        requests.post(os.environ.get("N8N_WEBHOOK_URL"), json=payload)



    @observe(as_type="generation")
    @tracer.wrap(name="transfer_call", service="outbound_calls")
    @function_tool()
    async def transfer_call(self, ctx: RunContext):
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

    # The voice agent will end the call when answering machine is detected (hasn't been tested)
    @observe(as_type="generation")
    @tracer.wrap(name="voicemall_detected", service="outbound_calls")
    @function_tool
    async def detected_answering_machine(self):
        """Call this tool if you have detected a voicemail system, AFTER hearing the voicemail greeting"""
        await self.session.generate_reply(
            instructions="Leave a voicemail message letting the user know you'll call back later."
        )
        await asyncio.sleep(0.5) # Add a natural gap to the end of the voicemail message
        await self.hangup()

    # End the call (doesn't work)
    @observe(as_type="generation")
    @tracer.wrap(name="end_call", service="outbound_calls")
    @function_tool()
    async def end_call(self, ctx: RunContext):
        """Called when the user wants to end the call"""
        logger.info(f"ending the call for {self.participant.identity}")

        # let the agent finish speaking
        current_speech = ctx.session.current_speech
        if current_speech:
            await current_speech.wait_for_playout()

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # use service key if backend
        client = supabase.create_client(supabase_url, supabase_key)

        # Serialize transcript into a single string or structured format
        transcript_text = "\n".join(f"{line['speaker']}: {line['text']}" for line in self.transcript)

        # Assuming your Supabase table is called "voice agent"
        client.table("voice agent").update({
        "transcript": transcript_text,
        "dateLastCalled": datetime.utcnow().isoformat()
        }).eq("phoneNumber", self.dial_info["phone_number"]).execute()

        await self.hangup()

    # For transcript - human speaker (hasn't been tested)
    @observe(as_type="generation")
    @tracer.wrap(name="on_user_turn_completed", service="outbound_calls")
    async def on_user_turn_completed(self, chat_ctx: llm.ChatContext, new_message: llm.ChatMessage):
        user_transcript = new_message.text_content
        logger.info(f"[User]: {user_transcript}")
        self.transcript.append({"speaker": "user", "text": user_transcript})

    # For transcript - voice agent (hasn't been tested)
    @observe(as_type="generation")
    @tracer.wrap(name="on_agent_turn_completed", service="outbound_calls")
    async def on_agent_turn_completed(self, chat_ctx: llm.ChatContext, new_message: llm.ChatMessage):
        agent_reply = new_message.text_content
        logger.info(f"[Agent]: {agent_reply}")
        self.transcript.append({"speaker": "agent", "text": agent_reply})

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
    participant_identity = phone_number = dial_info["phone_number"]

    # Create an MCP server - so livekit can use tools (e.g. GCal, Gmail)
    # print("Triggering URL:", os.environ.get("N8N_MCP_URL"))

    # mcp_server = MCPServerSse(
    #     params={"url": os.environ.get("N8N_MCP_URL")},
    #     cache_tools_list=True,
    #     name="SSE MCP Server"
    #     )

    # # Create an agent with MCP tools
    # agent = await MCPToolsIntegration.create_agent_with_tools(
    #         agent_class=OutboundCaller,
    #         agent_kwargs={
    #             "name": "Jayden",
    #             "appointment_time": "next Tuesday at 3pm",
    #             "dial_info": dial_info,
    #         },
    #         mcp_servers=[mcp_server]
    #     )

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
                # wait_until_answered=True,
            )
        )

        # wait for the agent session start and participant join
        await session_started
        participant = await ctx.wait_for_participant(identity=participant_identity)
        logger.info(f"participant joined: {participant.identity}")

        agent.set_participant(participant)

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
