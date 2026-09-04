import json
import os
import logging
import tempfile
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import anthropic
import caldav
import openai
from timezonefinder import TimezoneFinder
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

claude = anthropic.Anthropic()

MAX_HISTORY = 20
DEFAULT_TZ = ZoneInfo(os.environ.get("BOT_TIMEZONE", "America/Sao_Paulo"))
tf = TimezoneFinder()

NOTES_FILE = os.environ.get("NOTES_FILE", "/data/notes.json")

chat_histories: dict[int, list[dict]] = defaultdict(list)
chat_timezones: dict[int, ZoneInfo] = {}
pending_actions: dict[int, dict] = {}


def get_tz(chat_id: int) -> ZoneInfo:
    return chat_timezones.get(chat_id, DEFAULT_TZ)


def get_whisper():
    return openai.OpenAI()


def get_caldav_principal():
    client = caldav.DAVClient(
        url="https://caldav.icloud.com",
        username=os.environ["ICLOUD_EMAIL"],
        password=os.environ["ICLOUD_APP_PASSWORD"],
    )
    return client.principal()


def get_dropbox():
    import dropbox as dbx_lib
    refresh_token = os.environ.get("DROPBOX_REFRESH_TOKEN")
    if not refresh_token:
        raise RuntimeError(
            "Dropbox não configurado. Defina DROPBOX_APP_KEY, "
            "DROPBOX_APP_SECRET e DROPBOX_REFRESH_TOKEN."
        )
    return dbx_lib.Dropbox(
        oauth2_refresh_token=refresh_token,
        app_key=os.environ["DROPBOX_APP_KEY"],
        app_secret=os.environ["DROPBOX_APP_SECRET"],
    )


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes // 1024} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


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
- web_search: pesquisar na web por informações atualizadas ("pesquise X", "o que está acontecendo com Y")
- save_note: salvar uma nota organizada por assunto ("anote: ideia para app de receitas")
- list_notes: listar notas salvas ("quais minhas notas?", "notas sobre ideias")
- search_dropbox: pesquisar arquivos no Dropbox ("procure no Dropbox o arquivo X")
- list_dropbox: listar conteúdo de uma pasta do Dropbox ("o que tem na pasta Documents?")
- download_dropbox: baixar e enviar um arquivo do Dropbox ("me manda o arquivo X do Dropbox")

Quando pesquisar na web, organize um relatório claro e inclua as fontes no final.
Quando o usuário pedir para anotar algo, categorize automaticamente por assunto \
com base no conteúdo. Não peça confirmação para anotar — o comando já é a intenção.
Ao pesquisar no Dropbox, mostre os resultados com nome, caminho e tamanho. \
Para baixar, use download_dropbox com o caminho exato do arquivo encontrado.

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
    {
        "name": "save_note",
        "description": "Salva uma nota organizada por assunto. Use quando o usuário pedir para anotar algo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Assunto/categoria da nota (ex: ideias, trabalho, projetos)"},
                "text": {"type": "string", "description": "Conteúdo da nota"},
            },
            "required": ["subject", "text"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_notes",
        "description": "Lista notas salvas, opcionalmente filtradas por assunto.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Filtrar por assunto (opcional, se omitido lista todos)"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "search_dropbox",
        "description": "Pesquisa arquivos no Dropbox do usuário por nome ou conteúdo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Termo de busca"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_dropbox",
        "description": "Lista o conteúdo de uma pasta do Dropbox.",
        "input_schema": {
            "type": "object",
            "properties": {
                "folder_path": {
                    "type": "string",
                    "description": "Caminho da pasta (ex: /Documents). Use string vazia para a raiz.",
                },
            },
            "required": ["folder_path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "download_dropbox",
        "description": "Baixa um arquivo do Dropbox e envia ao usuário no Telegram.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Caminho completo do arquivo no Dropbox (ex: /Documents/report.pdf)",
                },
            },
            "required": ["file_path"],
            "additionalProperties": False,
        },
    },
]


def get_system_prompt(chat_id: int) -> str:
    tz = get_tz(chat_id)
    now = datetime.now(timezone.utc).astimezone(tz)
    today_str = f"{DAYS_PT[now.weekday()]}, {now.strftime('%d/%m/%Y')}"
    now_str = now.strftime("%H:%M")
    return SYSTEM_PROMPT_TEMPLATE.format(today=today_str, now=now_str)


def execute_list_events(start_date: str, end_date: str, chat_id: int) -> str:
    try:
        tz = get_tz(chat_id)
        principal = get_caldav_principal()
        calendars = principal.calendars()
        logger.info("Calendários encontrados: %d", len(calendars))

        start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=tz)
        end = datetime.strptime(end_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=tz
        )

        lines = []
        for cal in calendars:
            try:
                results = cal.search(start=start, end=end, event=True, expand=True)
            except Exception:
                logger.debug("Erro ao buscar em calendário, pulando")
                continue
            for r in results:
                ical = r.icalendar_instance
                for component in ical.walk():
                    if component.name == "VEVENT":
                        summary = str(component.get("SUMMARY", "Sem título"))
                        dtstart = component.get("DTSTART")
                        if dtstart and hasattr(dtstart.dt, "strftime"):
                            time_str = dtstart.dt.strftime("%d/%m/%Y %H:%M")
                        else:
                            time_str = str(dtstart.dt) if dtstart else "?"
                        lines.append(f"- {summary} ({time_str})")

        if not lines:
            return "Nenhum evento encontrado nesse período."

        return "\n".join(lines)
    except Exception as e:
        logger.exception("Erro ao listar eventos")
        return f"Erro ao consultar calendário: {e}"


def execute_create_event(
    title: str, start_datetime: str, chat_id: int, duration_minutes: int = 60
) -> str:
    try:
        tz = get_tz(chat_id)
        tz_name = str(tz)
        cal = get_caldav_principal().calendars()[0]
        start = datetime.strptime(start_datetime, "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=tz
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
            f"DTSTART;TZID={tz_name}:{start.strftime('%Y%m%dT%H%M%S')}\r\n"
            f"DTEND;TZID={tz_name}:{end.strftime('%Y%m%dT%H%M%S')}\r\n"
            "END:VEVENT\r\n"
            "END:VCALENDAR\r\n"
        )

        cal.save_event(ical_str)
        return f"Evento '{title}' criado para {start.strftime('%d/%m/%Y às %H:%M')}!"
    except Exception as e:
        logger.exception("Erro ao criar evento")
        return f"Erro ao criar evento: {e}"


def load_notes() -> dict:
    try:
        with open(NOTES_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_notes_to_file(notes: dict) -> None:
    os.makedirs(os.path.dirname(NOTES_FILE) or ".", exist_ok=True)
    with open(NOTES_FILE, "w") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)


def execute_save_note(subject: str, text: str, chat_id: int) -> str:
    try:
        tz = get_tz(chat_id)
        now = datetime.now(timezone.utc).astimezone(tz)
        notes = load_notes()
        if subject not in notes:
            notes[subject] = []
        notes[subject].append({
            "text": text,
            "timestamp": now.strftime("%d/%m/%Y %H:%M"),
        })
        save_notes_to_file(notes)
        total = len(notes[subject])
        return f"Nota salva em '{subject}' ({total} nota(s) nesse assunto)."
    except Exception as e:
        logger.exception("Erro ao salvar nota")
        return f"Erro ao salvar nota: {e}"


def execute_list_notes(subject: str = None) -> str:
    try:
        notes = load_notes()
        if not notes:
            return "Nenhuma nota salva ainda."
        if subject:
            items = notes.get(subject)
            if not items:
                subjects = ", ".join(notes.keys())
                return f"Nenhuma nota sobre '{subject}'. Assuntos disponíveis: {subjects}"
            lines = [f"Notas sobre '{subject}':"]
            for item in items:
                lines.append(f"- {item['text']} ({item['timestamp']})")
            return "\n".join(lines)
        lines = []
        for subj, items in notes.items():
            lines.append(f"\n{subj} ({len(items)}):")
            for item in items:
                lines.append(f"  - {item['text']} ({item['timestamp']})")
        return "Suas notas:" + "\n".join(lines)
    except Exception as e:
        logger.exception("Erro ao listar notas")
        return f"Erro ao listar notas: {e}"


def execute_search_dropbox(query: str) -> str:
    try:
        dbx = get_dropbox()
        result = dbx.files_search_v2(query)
        if not result.matches:
            return f"Nenhum arquivo encontrado para '{query}'."
        lines = []
        for match in result.matches[:15]:
            metadata = match.metadata.get_metadata()
            name = metadata.name
            path = metadata.path_display
            size = getattr(metadata, "size", None)
            size_str = f" ({_format_size(size)})" if size else ""
            lines.append(f"- {name}{size_str}\n  {path}")
        total = len(result.matches)
        if total > 15:
            lines.append(f"\n... e mais {total - 15} resultados")
        return "\n".join(lines)
    except Exception as e:
        logger.exception("Erro ao pesquisar Dropbox")
        return f"Erro ao pesquisar no Dropbox: {e}"


def execute_list_dropbox(folder_path: str) -> str:
    try:
        dbx = get_dropbox()
        path = folder_path if folder_path != "/" else ""
        result = dbx.files_list_folder(path)
        if not result.entries:
            return f"Pasta '{folder_path}' está vazia."
        lines = []
        for entry in result.entries:
            size = getattr(entry, "size", None)
            if size is not None:
                lines.append(f"- {entry.name} ({_format_size(size)})\n  {entry.path_display}")
            else:
                lines.append(f"- {entry.name}/\n  {entry.path_display}")
        if result.has_more:
            lines.append(f"\n... e mais arquivos")
        return "\n".join(lines)
    except Exception as e:
        logger.exception("Erro ao listar pasta Dropbox")
        return f"Erro ao listar pasta: {e}"


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
    elif action["type"] == "download":
        d = action["data"]
        return (
            f"Enviar este arquivo do Dropbox?\n\n"
            f"{d['file_path']}\n\n"
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


WEB_TOOLS = [
    {"type": "web_search_20260209", "name": "web_search"},
]

MAX_MSG_LEN = 4096


def _extract_response(response):
    tool_use_block = None
    text_parts = []
    citations = []
    for block in response.content:
        if block.type == "tool_use":
            tool_use_block = block
        elif block.type == "text":
            text_parts.append(block.text)
            if hasattr(block, "citations") and block.citations:
                for cite in block.citations:
                    url = getattr(cite, "url", None)
                    if url:
                        citations.append({
                            "url": url,
                            "title": getattr(cite, "title", ""),
                        })
    return tool_use_block, text_parts, citations


def _format_sources(citations: list[dict]) -> str:
    if not citations:
        return ""
    seen = set()
    sources = []
    for c in citations:
        if c["url"] not in seen:
            seen.add(c["url"])
            title = c["title"] or c["url"]
            sources.append(f"- {title}\n  {c['url']}")
    return "\n\nFontes:\n" + "\n".join(sources)


def ask_claude(chat_id: int, user_text: str) -> tuple[str, dict | None]:
    history = chat_histories[chat_id]
    history.append({"role": "user", "content": user_text})

    if len(history) > MAX_HISTORY:
        history[:] = history[-MAX_HISTORY:]

    system_prompt = get_system_prompt(chat_id)
    all_tools = TOOLS + WEB_TOOLS

    response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16000,
        system=system_prompt,
        tools=all_tools,
        messages=history,
    )

    all_text = []
    all_citations = []
    restarts = 0
    while response.stop_reason == "pause_turn" and restarts < 5:
        for block in response.content:
            if block.type == "text":
                all_text.append(block.text)
                if hasattr(block, "citations") and block.citations:
                    for cite in block.citations:
                        url = getattr(cite, "url", None)
                        if url:
                            all_citations.append({"url": url, "title": getattr(cite, "title", "")})
        resume_messages = list(history) + [
            {"role": "assistant", "content": response.content},
        ]
        response = claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=16000,
            system=system_prompt,
            tools=all_tools,
            messages=resume_messages,
        )
        restarts += 1

    tool_use_block, text_parts, citations = _extract_response(response)
    all_text.extend(text_parts)
    all_citations.extend(citations)

    immediate_tools = {"list_events", "save_note", "list_notes", "search_dropbox", "list_dropbox"}
    if tool_use_block and tool_use_block.name in immediate_tools:
        if tool_use_block.name == "list_events":
            result = execute_list_events(**tool_use_block.input, chat_id=chat_id)
        elif tool_use_block.name == "save_note":
            result = execute_save_note(**tool_use_block.input, chat_id=chat_id)
        elif tool_use_block.name == "search_dropbox":
            result = execute_search_dropbox(**tool_use_block.input)
        elif tool_use_block.name == "list_dropbox":
            result = execute_list_dropbox(**tool_use_block.input)
        else:
            result = execute_list_notes(**tool_use_block.input)

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
            max_tokens=16000,
            system=system_prompt,
            tools=all_tools,
            messages=followup_messages,
        )

        _, final_text, final_citations = _extract_response(final_response)
        reply = "".join(final_text) or result
        reply += _format_sources(final_citations)

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

    elif tool_use_block and tool_use_block.name == "download_dropbox":
        action = {"type": "download", "data": tool_use_block.input}
        reply = format_confirm(action)
        history.append({"role": "assistant", "content": reply})
        if len(history) > MAX_HISTORY:
            history[:] = history[-MAX_HISTORY:]
        return reply, action

    reply = "".join(all_text) or "Sem resposta."
    reply += _format_sources(all_citations)
    history.append({"role": "assistant", "content": reply})
    if len(history) > MAX_HISTORY:
        history[:] = history[-MAX_HISTORY:]
    return reply, None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    chat_histories[chat_id].clear()
    pending_actions.pop(chat_id, None)
    logger.info("/start de %s", update.effective_user.first_name)
    tz = get_tz(chat_id)
    await update.message.reply_text(
        "Olá! Eu sou o Barca, seu assistente com IA.\n"
        "Mande texto ou áudio. Posso consultar/criar eventos e agendar lembretes.\n\n"
        f"Fuso horário atual: {tz}\n"
        "Para ajustar, envie sua localização (clipe > Localização)."
    )


async def handle_location(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    lat = update.message.location.latitude
    lng = update.message.location.longitude
    tz_name = tf.timezone_at(lat=lat, lng=lng)
    if tz_name:
        chat_timezones[chat_id] = ZoneInfo(tz_name)
        now = datetime.now(timezone.utc).astimezone(chat_timezones[chat_id])
        await update.message.reply_text(
            f"Fuso horário atualizado: {tz_name}\n"
            f"Hora local: {now.strftime('%d/%m/%Y %H:%M')}"
        )
        logger.info("Timezone de chat %s atualizado para %s", chat_id, tz_name)
    else:
        await update.message.reply_text("Não consegui detectar o fuso horário dessa localização.")


async def execute_confirmed_action(
    action: dict, chat_id: int, context: ContextTypes.DEFAULT_TYPE
) -> str:
    tz = get_tz(chat_id)
    if action["type"] == "event":
        d = action["data"]
        return execute_create_event(
            title=d["title"],
            start_datetime=d["start_datetime"],
            chat_id=chat_id,
            duration_minutes=d.get("duration_minutes", 60),
        )
    else:
        d = action["data"]
        remind_at = datetime.strptime(d["remind_at"], "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=tz
        )
        now = datetime.now(timezone.utc).astimezone(tz)
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


async def execute_download(action: dict, chat_id: int, update: Update) -> str:
    try:
        file_path = action["data"]["file_path"]
        dbx = get_dropbox()
        metadata, res = dbx.files_download(file_path)
        ext = os.path.splitext(file_path)[1] or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(res.content)
            tmp_path = tmp.name
        try:
            with open(tmp_path, "rb") as f:
                await update.message.reply_document(document=f, filename=metadata.name)
        finally:
            os.unlink(tmp_path)
        return f"Arquivo '{metadata.name}' enviado!"
    except Exception as e:
        logger.exception("Erro ao baixar do Dropbox")
        return f"Erro ao baixar arquivo: {e}"


async def send_reply(update: Update, text: str) -> None:
    for i in range(0, len(text), MAX_MSG_LEN):
        await update.message.reply_text(text[i:i + MAX_MSG_LEN])


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
                if action["type"] == "download":
                    result = await execute_download(action, chat_id, update)
                else:
                    result = await execute_confirmed_action(action, chat_id, context)
                history.append({"role": "assistant", "content": result})
                if action["type"] != "download" or result.startswith("Erro"):
                    await send_reply(update, result)
            else:
                cancel = "Ok, cancelado."
                history.append({"role": "assistant", "content": cancel})
                await update.message.reply_text(cancel)
            return

        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        reply, pending = ask_claude(chat_id, user_text)
        if pending:
            pending_actions[chat_id] = pending
        await send_reply(update, reply)
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

        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        reply, pending = ask_claude(chat_id, user_text)
        if pending:
            pending_actions[chat_id] = pending
        await send_reply(update, reply)
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
    app.add_handler(MessageHandler(filters.LOCATION, handle_location))
    now = datetime.now(timezone.utc).astimezone(DEFAULT_TZ)
    logger.info("Bot iniciado. Data/hora padrão (%s): %s", DEFAULT_TZ, now.strftime("%d/%m/%Y %H:%M:%S"))
    logger.info("Bot iniciado (long polling)...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
