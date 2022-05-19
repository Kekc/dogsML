import telegram
import telegram.ext
import dogsml.tfbased.predict


def start(
    update: telegram.Update,
    context: telegram.ext.CallbackContext
):
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Dogs ML bot"
    )


def image_function(
    update: telegram.Update,
    context: telegram.ext.CallbackContext
):
    image_id = update["message"]["photo"][-1]["file_id"]
    image_file = context.bot.get_file(image_id)
    p_value = dogsml.tfbased.predict.predict_from_url(image_file.file_path)

    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=dogsml.tfbased.predict.interpret_predicted_dog_values(p_value)
    )


def unknown(
    update: telegram.Update,
    context: telegram.ext.CallbackContext
):
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Unknown command."
    )
