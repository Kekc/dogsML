import logging
import telegram
import telegram.ext

import dogsml.settings
import dogsml.tfbased.predict

import handlers


updater = telegram.ext.Updater(token=dogsml.settings.TELEGRAM_BOT_TOKEN)
dispatcher = updater.dispatcher

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


start_handler = telegram.ext.CommandHandler(
    "start",
    handlers.start,
)
image_handler = telegram.ext.MessageHandler(
    ~telegram.ext.Filters.command,
    handlers.image_function,
)
unknown_handler = telegram.ext.MessageHandler(
    telegram.ext.Filters.command,
    handlers.unknown,
)

dispatcher.add_handler(start_handler)
dispatcher.add_handler(image_handler)
dispatcher.add_handler(unknown_handler)


def run_bot():
    updater.start_polling()


if __name__ == "__main__":
    run_bot()
