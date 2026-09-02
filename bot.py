import os
import logging
import tempfile

import anthropic
import openai
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

claude = anthropic.Anthropic()


def get_whisper():
    return openai.OpenAI()

SYSTEM_PROMPT = """\
Você é o Barca, um assistente inteligente no Telegram. \
Responda de forma útil, clara e concisa em português do Brasil.

Regras obrigatórias:
- Leitura é livre; qualquer ação de escrita, exclusão, movimentação ou envio \
exige confirmação explícita do usuário antes de executar.
- Nunca apague dados — apenas adicione (ex.: crie eventos, nunca delete).
- Nunca manipule dinheiro, senhas, tokens ou credenciais.
- Se você não sabe ou não consegue fazer algo, diga honestamente — nunca invente.\
"""


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("/start de %s", update.effective_user.first_name)
    await update.message.reply_text(
        "Olá! Eu sou o Barca, seu assistente com IA. Mande qualquer mensagem!"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = update.message.text
    logger.info("Mensagem de %s: %s", update.effective_user.first_name, user_text)

    try:
        response = claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_text}],
        )

        reply = ""
        for block in response.content:
            if block.type == "text":
                reply += block.text

        await update.message.reply_text(reply or "Sem resposta.")

    except Exception as e:
        logger.exception("Erro ao processar mensagem")
        await update.message.reply_text(f"Erro: {e}")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

        response = claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_text}],
        )

        reply = ""
        for block in response.content:
            if block.type == "text":
                reply += block.text

        await update.message.reply_text(reply or "Sem resposta.")

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
