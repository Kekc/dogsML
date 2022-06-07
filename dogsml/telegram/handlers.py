import telegram
import telegram.ext
import dogsml.tfbased.predict
import dogsml.transfer.constants


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
    dog_breed_model = context.bot_data.get("dog_breed_model")
    if not dog_breed_model:
        dog_breed_model = dogsml.tfbased.predict.load_model("dog_breeds_5.h5")
        context.bot_data["dog_breed_model"] = dog_breed_model

    p_value = dogsml.tfbased.predict.predict_from_url(
        image_file.file_path,
        dog_breed_model,
        224,
        224,
    )
    probabilities = dogsml.tfbased.predict.interpret_result_with_probabilities(
        p_value,
        dogsml.transfer.constants.DOG_BREEDS,
        3,
    )
    message = ""
    for label, value in probabilities.items():
        printed_value = format(value * 100, ".2f")
        message += "{0} --- {1}%\n".format(label, printed_value)
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=message,
    )


def unknown(
    update: telegram.Update,
    context: telegram.ext.CallbackContext
):
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Unknown command."
    )
