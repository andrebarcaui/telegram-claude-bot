import os
import logging
import tempfile
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import anthropic
import caldav
import openai
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

claude = anthropic.Anthropic()

MAX_HISTORY = 20
TZ = ZoneInfo("America/Sao_Paulo")

chat_histories: dict[int, list[dict]] = defaultdict(list)
pending_events: dict[int, dict] = {}


def get_whisper():
    return openai.OpenAI()


def get_calendar():
    client = caldav.DAVClient(
        url="https://caldav.icloud.com",
        username=os.environ["ICLOUD_EMAIL"],
        password=os.environ["ICLOUD_APP_PASSWORD"],
    )
    return client.principal().calendars()[0]


DAYS_PT = [
    "segunda-feira", "terça-feira", "quarta-feira",
    "quinta-feira", "sexta-feira", "sábado", "domingo",
]

SYSTEM_PROMPT_TEMPLATE = """\
Você é o Barca, um assistente inteligente no Telegram. \
Responda de forma útil, clara e concisa em português do Brasil.
Hoje é {today}. Use essa data como referência para "amanhã", "sexta", etc.

Você tem acesso ao calendário do usuário via ferramentas. \
Use list_events para consultar e create_event para criar eventos.

Regras obrigatórias:
- Leitura é livre; qualquer ação de escrita, exclusão, movimentação ou envio \
exige confirmação explícita do usuário antes de executar.
- Nunca apague dados — apenas adicione (ex.: crie eventos, nunca delete).
- Nunca manipule dinheiro, senhas, tokens ou credenciais.
- Se você não sabe ou não consegue fazer algo, diga honestamente — nunca invente.\
"""

CALENDAR_TOOLS = [
    {
        "name": "list_events",
        "description": "Lista eventos do calendário em um período. Use para consultas como 'o que tenho amanhã?', 'minha agenda da semana', etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "Data de início (YYYY-MM-DD)"},
                "end_date": {"type": "string", "description": "Data de fim (YYYY-MM-DD)"},
            },
            "required": ["start_date", "end_date"],
            "additionalProperties": False,
        },
    },
    {
        "name": "create_event",
        "description": "Cria um novo evento no calendário. Use para 'marcar dentista sexta 15h', 'agendar reunião amanhã às 10h', etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Título do evento"},
                "start_datetime": {"type": "string", "description": "Início (YYYY-MM-DDTHH:MM:SS)"},
                "duration_minutes": {"type": "integer", "description": "Duração em minutos (padrão 60)"},
            },
            "required": ["title", "start_datetime"],
            "additionalProperties": False,
        },
    },
]


def get_system_prompt() -> str:
    now = datetime.now(TZ)
    today_str = f"{DAYS_PT[now.weekday()]}, {now.strftime('%d/%m/%Y')}"
    return SYSTEM_PROMPT_TEMPLATE.format(today=today_str)


def execute_list_events(start_date: str, end_date: str) -> str:
    try:
        cal = get_calendar()
        start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=TZ)
        end = datetime.strptime(end_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=TZ
        )
        results = cal.search(start=start, end=end, event=True, expand=True)

        if not results:
            return "Nenhum evento encontrado nesse período."

        lines = []
        for r in results:
            vevent = r.vobject_instance.vevent
            summary = str(vevent.summary.value)
            dtstart = vevent.dtstart.value
            if hasattr(dtstart, "strftime"):
                time_str = dtstart.strftime("%d/%m/%Y %H:%M")
            else:
                time_str = str(dtstart)
            lines.append(f"- {summary} ({time_str})")

        return "\n".join(lines)
    except Exception as e:
        logger.exception("Erro ao listar eventos")
        return f"Erro ao consultar calendário: {e}"


def execute_create_event(
    title: str, start_datetime: str, duration_minutes: int = 60
) -> str:
    try:
        cal = get_calendar()
        start = datetime.strptime(start_datetime, "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=TZ
        )
        end = start + timedelta(minutes=duration_minutes)
        uid = str(uuid.uuid4())

        ical_str = (
            "BEGIN:VCALENDAR\r\n"
            "VERSION:2.0\r\n"
            "PRODID:-//Barca Bot//PT\r\n"
            "BEGIN:VEVENT\r\n"
            f"UID:{uid}\r\n"
            f"SUMMARY:{title}\r\n"
            f"DTSTART;TZID=America/Sao_Paulo:{start.strftime('%Y%m%dT%H%M%S')}\r\n"
            f"DTEND;TZID=America/Sao_Paulo:{end.strftime('%Y%m%dT%H%M%S')}\r\n"
            "END:VEVENT\r\n"
            "END:VCALENDAR\r\n"
        )

        cal.save_event(ical_str)
        return f"Evento '{title}' criado com sucesso para {start.strftime('%d/%m/%Y às %H:%M')}!"
    except Exception as e:
        logger.exception("Erro ao criar evento")
        return f"Erro ao criar evento: {e}"


def ask_claude(chat_id: int, user_text: str) -> tuple[str, dict | None]:
    history = chat_histories[chat_id]
    history.append({"role": "user", "content": user_text})

    if len(history) > MAX_HISTORY:
        history[:] = history[-MAX_HISTORY:]

    system_prompt = get_system_prompt()

    response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=system_prompt,
        tools=CALENDAR_TOOLS,
        messages=history,
    )

    tool_use_block = None
    text_parts = []
    for block in response.content:
        if block.type == "tool_use":
            tool_use_block = block
        elif block.type == "text":
            text_parts.append(block.text)

    if tool_use_block and tool_use_block.name == "list_events":
        result = execute_list_events(**tool_use_block.input)

        followup_messages = list(history) + [
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_block.id,
                        "content": result,
                    }
                ],
            },
        ]

        final_response = claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=system_prompt,
            tools=CALENDAR_TOOLS,
            messages=followup_messages,
        )

        reply = ""
        for block in final_response.content:
            if block.type == "text":
                reply += block.text
        reply = reply or result

        history.append({"role": "assistant", "content": reply})
        if len(history) > MAX_HISTORY:
            history[:] = history[-MAX_HISTORY:]
        return reply, None

    elif tool_use_block and tool_use_block.name == "create_event":
        event_data = tool_use_block.input
        title = event_data.get("title", "")
        start_dt = event_data.get("start_datetime", "")
        duration = event_data.get("duration_minutes", 60)

        try:
            dt = datetime.strptime(start_dt, "%Y-%m-%dT%H:%M:%S")
            formatted = dt.strftime("%d/%m/%Y às %H:%M")
        except ValueError:
            formatted = start_dt

        reply = (
            f"Posso criar este evento?\n\n"
            f"{title}\n"
            f"{formatted}\n"
            f"Duração: {duration} min\n\n"
            f"Confirma? (sim/não)"
        )

        history.append({"role": "assistant", "content": reply})
        if len(history) > MAX_HISTORY:
            history[:] = history[-MAX_HISTORY:]
        return reply, event_data

    reply = "".join(text_parts) or "Sem resposta."
    history.append({"role": "assistant", "content": reply})
    if len(history) > MAX_HISTORY:
        history[:] = history[-MAX_HISTORY:]
    return reply, None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    chat_histories[chat_id].clear()
    pending_events.pop(chat_id, None)
    logger.info("/start de %s", update.effective_user.first_name)
    await update.message.reply_text(
        "Olá! Eu sou o Barca, seu assistente com IA.\n"
        "Mande texto ou áudio. Posso consultar e criar eventos no seu calendário."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = update.message.text
    chat_id = update.effective_chat.id
    logger.info("Mensagem de %s: %s", update.effective_user.first_name, user_text)

    try:
        if chat_id in pending_events:
            event_data = pending_events.pop(chat_id)
            lower = user_text.lower().strip()
            if lower in ("sim", "s", "yes", "y", "ok", "pode", "confirma"):
                result = execute_create_event(
                    title=event_data["title"],
                    start_datetime=event_data["start_datetime"],
                    duration_minutes=event_data.get("duration_minutes", 60),
                )
                history = chat_histories[chat_id]
                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": result})
                await update.message.reply_text(result)
            else:
                history = chat_histories[chat_id]
                history.append({"role": "user", "content": user_text})
                cancel = "Ok, evento não criado."
                history.append({"role": "assistant", "content": cancel})
                await update.message.reply_text(cancel)
            return

        reply, pending = ask_claude(chat_id, user_text)
        if pending:
            pending_events[chat_id] = pending
        await update.message.reply_text(reply)
    except Exception as e:
        logger.exception("Erro ao processar mensagem")
        await update.message.reply_text(f"Erro: {e}")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    logger.info("Áudio de %s", update.effective_user.first_name)

    try:
        voice_file = await update.message.voice.get_file()
        with tempfile.NamedTemporaryFile(suffix=".ogg") as tmp:
            await voice_file.download_to_drive(tmp.name)
            with open(tmp.name, "rb") as audio:
                transcription = get_whisper().audio.transcriptions.create(
                    model="whisper-1", file=audio
                )

        user_text = transcription.text
        logger.info("Transcrição: %s", user_text)

        reply, pending = ask_claude(chat_id, user_text)
        if pending:
            pending_events[chat_id] = pending
        await update.message.reply_text(reply)
    except Exception as e:
        logger.exception("Erro ao processar áudio")
        await update.message.reply_text(f"Erro ao processar áudio: {e}")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Exceção não tratada:", exc_info=context.error)


def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = ApplicationBuilder().token(token).build()
    app.add_error_handler(error_handler)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    logger.info("Bot iniciado (long polling)...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
