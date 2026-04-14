class IntentClassification:
    def __init__(self, model_path):
        self.model_path = model_path

    def __call__(self, message):
        raise NotImplementedError("Inference logic will be implemented in the next stage.")


if __name__ == "__main__":
    classifier = IntentClassification("configs/inference.yaml")
    example_message = "My card has not arrived yet."
    print(classifier(example_message))
