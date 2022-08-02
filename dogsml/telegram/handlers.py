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


def text_function(
    update: telegram.Update,
    context: telegram.ext.CallbackContext
):
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Send picture of a dog"
    )


def help_function(
    update: telegram.Update,
    context: telegram.ext.CallbackContext
):
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=("This is a Telegram bot for dog's breed classification."
              "Send a picture of a dog and it will return three most"
              "possible breeds.")
    )


def image_function(
    update: telegram.Update,
    context: telegram.ext.CallbackContext
):
    image_id = update["message"]["photo"][-1]["file_id"]
    image_file = context.bot.get_file(image_id)
    dog_breed_model = context.bot_data.get("dog_breed_model")
    if not dog_breed_model:
        dog_breed_model = dogsml.tfbased.predict.load_model(
            "dog_breed_colab.h5"
        )
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
    sorted_probabilities = sorted(
        ((v, k) for k, v in probabilities.items()),
        reverse=True
    )
    for value, label in sorted_probabilities:
        printed_value = format(value * 100, ".2f")
        message += "{0} --- {1}%\n".format(label, printed_value)
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=message,
    )
