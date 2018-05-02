import os
import npyscreen
import tensorflow as tf
from useNetwork import use_neural_network
tf.logging.set_verbosity(tf.logging.ERROR)


class TestApp(npyscreen.NPSApp):
    text = None
    getSentimentButton = None
    sentiment = None

    def main(self):
        form = npyscreen.Form(name="Sentiment Analysis",)
        self.text = form.add(npyscreen.TitleText, name="Text:",)
        self.getSentimentButton = form.add(npyscreen.ButtonPress, name="Get Sentiment")
        self.getSentimentButton.when_pressed_function = self.displaySentiment
        self.sentiment = form.add(npyscreen.TitleSelectOne, max_height=4, editable=False, name="Sentiment:",
                                  values=["Happy", "Sad"], scroll_exit=True)
        form.edit()

    def displaySentiment(self):
        self.text.editable = False
        self.getSentimentButton.editable = False
        self.sentiment.value = [use_neural_network(self.text.value) - 1]


if __name__ == "__main__":
    App = TestApp()
    App.run()
