import os
import logging

import anthropic
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

claude = anthropic.Anthropic()

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


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = update.message.text
    logger.info("Mensagem de %s: %s", update.effective_user.first_name, user_text)

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


def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = ApplicationBuilder().token(token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Bot iniciado (long polling)...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
