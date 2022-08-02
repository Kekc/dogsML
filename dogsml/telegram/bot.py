import logging
import telegram
import telegram.ext

import dogsml.settings
import dogsml.tfbased.predict

import handlers


updater = telegram.ext.Updater(token=dogsml.settings.TELEGRAM_BOT_TOKEN)
dispatcher = updater.dispatcher


start_handler = telegram.ext.CommandHandler(
    "start",
    handlers.start,
)
help_handler = telegram.ext.CommandHandler(
    "help",
    handlers.help_function,
)

text_handler = telegram.ext.MessageHandler(
    telegram.ext.Filters.text,
    handlers.text_function,
)
image_handler = telegram.ext.MessageHandler(
    ~telegram.ext.Filters.command,
    handlers.image_function,
)

dispatcher.add_handler(start_handler)
dispatcher.add_handler(help_handler)
dispatcher.add_handler(text_handler)
dispatcher.add_handler(image_handler)


def run_bot():
    logging.basicConfig(
        filename="example.log",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    updater.start_polling()


if __name__ == "__main__":
    run_bot()
