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
pending_actions: dict[int, dict] = {}


def get_whisper():
    return openai.OpenAI()


def get_caldav_principal():
    client = caldav.DAVClient(
        url="https://caldav.icloud.com",
        username=os.environ["ICLOUD_EMAIL"],
        password=os.environ["ICLOUD_APP_PASSWORD"],
    )
    return client.principal()


DAYS_PT = [
    "segunda-feira", "terça-feira", "quarta-feira",
    "quinta-feira", "sexta-feira", "sábado", "domingo",
]

SYSTEM_PROMPT_TEMPLATE = """\
Você é o Barca, um assistente inteligente no Telegram. \
Responda de forma útil, clara e concisa em português do Brasil.
Hoje é {today} ({now}). Use como referência para "amanhã", "sexta", etc.

Ferramentas disponíveis:
- list_events: consultar agenda ("o que tenho amanhã?")
- create_event: criar evento no calendário ("marcar dentista sexta 15h")
- set_reminder: agendar lembrete no Telegram ("me lembra às 6h30 de segunda de X")

Regras obrigatórias:
- Leitura é livre; qualquer ação de escrita, exclusão, movimentação ou envio \
exige confirmação explícita do usuário antes de executar.
- Nunca apague dados — apenas adicione (ex.: crie eventos, nunca delete).
- Nunca manipule dinheiro, senhas, tokens ou credenciais.
- Se você não sabe ou não consegue fazer algo, diga honestamente — nunca invente.\
"""

TOOLS = [
    {
        "name": "list_events",
        "description": "Lista eventos do calendário em um período.",
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
        "description": "Cria um novo evento no calendário.",
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
    {
        "name": "set_reminder",
        "description": "Agenda um lembrete que será enviado no Telegram na data/hora especificada.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reminder_text": {"type": "string", "description": "Texto do lembrete"},
                "remind_at": {"type": "string", "description": "Data e hora do lembrete (YYYY-MM-DDTHH:MM:SS)"},
            },
            "required": ["reminder_text", "remind_at"],
            "additionalProperties": False,
        },
    },
]


def get_system_prompt() -> str:
    now = datetime.now(TZ)
    today_str = f"{DAYS_PT[now.weekday()]}, {now.strftime('%d/%m/%Y')}"
    now_str = now.strftime("%H:%M")
    return SYSTEM_PROMPT_TEMPLATE.format(today=today_str, now=now_str)


def execute_list_events(start_date: str, end_date: str) -> str:
    try:
        principal = get_caldav_principal()
        calendars = principal.calendars()
        logger.info("Calendários encontrados: %s", [c.name for c in calendars])

        start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=TZ)
        end = datetime.strptime(end_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=TZ
        )

        lines = []
        for cal in calendars:
            results = cal.search(start=start, end=end, event=True, expand=True)
            for r in results:
                vevent = r.vobject_instance.vevent
                summary = str(vevent.summary.value)
                dtstart = vevent.dtstart.value
                if hasattr(dtstart, "strftime"):
                    time_str = dtstart.strftime("%d/%m/%Y %H:%M")
                else:
                    time_str = str(dtstart)
                lines.append(f"- {summary} ({time_str})")

        if not lines:
            return "Nenhum evento encontrado nesse período."

        return "\n".join(lines)
    except Exception as e:
        logger.exception("Erro ao listar eventos")
        return f"Erro ao consultar calendário: {e}"


def execute_create_event(
    title: str, start_datetime: str, duration_minutes: int = 60
) -> str:
    try:
        cal = get_caldav_principal().calendars()[0]
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
        return f"Evento '{title}' criado para {start.strftime('%d/%m/%Y às %H:%M')}!"
    except Exception as e:
        logger.exception("Erro ao criar evento")
        return f"Erro ao criar evento: {e}"


async def fire_reminder(context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = context.job.chat_id
    text = context.job.data
    logger.info("Disparando lembrete para chat %s: %s", chat_id, text)
    await context.bot.send_message(chat_id=chat_id, text=f"Lembrete: {text}")


def format_confirm(action: dict) -> str:
    if action["type"] == "event":
        d = action["data"]
        try:
            dt = datetime.strptime(d["start_datetime"], "%Y-%m-%dT%H:%M:%S")
            formatted = dt.strftime("%d/%m/%Y às %H:%M")
        except ValueError:
            formatted = d["start_datetime"]
        return (
            f"Posso criar este evento?\n\n"
            f"{d['title']}\n"
            f"{formatted}\n"
            f"Duração: {d.get('duration_minutes', 60)} min\n\n"
            f"Confirma? (sim/não)"
        )
    else:
        d = action["data"]
        try:
            dt = datetime.strptime(d["remind_at"], "%Y-%m-%dT%H:%M:%S")
            formatted = dt.strftime("%d/%m/%Y às %H:%M")
        except ValueError:
            formatted = d["remind_at"]
        return (
            f"Agendar este lembrete?\n\n"
            f"{d['reminder_text']}\n"
            f"{formatted}\n\n"
            f"Confirma? (sim/não)"
        )


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
        tools=TOOLS,
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
            tools=TOOLS,
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
        action = {"type": "event", "data": tool_use_block.input}
        reply = format_confirm(action)
        history.append({"role": "assistant", "content": reply})
        if len(history) > MAX_HISTORY:
            history[:] = history[-MAX_HISTORY:]
        return reply, action

    elif tool_use_block and tool_use_block.name == "set_reminder":
        action = {"type": "reminder", "data": tool_use_block.input}
        reply = format_confirm(action)
        history.append({"role": "assistant", "content": reply})
        if len(history) > MAX_HISTORY:
            history[:] = history[-MAX_HISTORY:]
        return reply, action

    reply = "".join(text_parts) or "Sem resposta."
    history.append({"role": "assistant", "content": reply})
    if len(history) > MAX_HISTORY:
        history[:] = history[-MAX_HISTORY:]
    return reply, None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    chat_histories[chat_id].clear()
    pending_actions.pop(chat_id, None)
    logger.info("/start de %s", update.effective_user.first_name)
    await update.message.reply_text(
        "Olá! Eu sou o Barca, seu assistente com IA.\n"
        "Mande texto ou áudio. Posso consultar/criar eventos e agendar lembretes."
    )


async def execute_confirmed_action(
    action: dict, chat_id: int, context: ContextTypes.DEFAULT_TYPE
) -> str:
    if action["type"] == "event":
        d = action["data"]
        return execute_create_event(
            title=d["title"],
            start_datetime=d["start_datetime"],
            duration_minutes=d.get("duration_minutes", 60),
        )
    else:
        d = action["data"]
        remind_at = datetime.strptime(d["remind_at"], "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=TZ
        )
        now = datetime.now(TZ)
        delay = (remind_at - now).total_seconds()
        if delay <= 0:
            return "Esse horário já passou. Escolha um horário futuro."
        context.job_queue.run_once(
            fire_reminder,
            when=delay,
            chat_id=chat_id,
            data=d["reminder_text"],
        )
        return f"Lembrete agendado para {remind_at.strftime('%d/%m/%Y às %H:%M')}!"


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = update.message.text
    chat_id = update.effective_chat.id
    logger.info("Mensagem de %s: %s", update.effective_user.first_name, user_text)

    try:
        if chat_id in pending_actions:
            action = pending_actions.pop(chat_id)
            lower = user_text.lower().strip()
            history = chat_histories[chat_id]
            history.append({"role": "user", "content": user_text})
            if lower in ("sim", "s", "yes", "y", "ok", "pode", "confirma"):
                result = await execute_confirmed_action(action, chat_id, context)
                history.append({"role": "assistant", "content": result})
                await update.message.reply_text(result)
            else:
                cancel = "Ok, cancelado."
                history.append({"role": "assistant", "content": cancel})
                await update.message.reply_text(cancel)
            return

        reply, pending = ask_claude(chat_id, user_text)
        if pending:
            pending_actions[chat_id] = pending
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
            pending_actions[chat_id] = pending
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
    now = datetime.now(TZ)
    logger.info("Bot iniciado. Data/hora São Paulo: %s", now.strftime("%d/%m/%Y %H:%M:%S"))
    logger.info("Bot iniciado (long polling)...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
