import os
import spacy
from spacy import displacy
from typing import Optional, Union, List, Dict

class EntityRecognizer:
    """
    A class that performs entity recognition using a specified model.

    Args:
        model_path (Union[str, os.PathLike]): The path to the entity recognition model.
        doc (Optional[str]): The optional document associated with the entity recognition.

    Attributes:
        model_path (Union[str, os.PathLike]): The path to the entity recognition model.
        model (spacy.language.Language): The loaded entity recognition model.
        COLORS (Dict[str, str]): A dictionary mapping entity types to colors for visualization.
        doc (Optional[spacy.tokens.doc.Doc]): The processed document associated with the entity recognition.

    Methods:
        predict(text: str) -> List[Dict[str, Union[str, int, Dict[str, Union[str, int]]]]]:
            Performs entity recognition on the given text and returns the detected entities.

        display(text: str) -> None:
            Displays the entity recognition visualization for the given text.
    """

    def __init__(self, model_path: Union[str, os.PathLike], doc: Optional[str] = None):
        """
        Initialize the EntityRecognizer object.

        Args:
            model_path (Union[str, os.PathLike]): The path to the entity recognition model.
            doc (Optional[str]): The optional document associated with the entity recognition.
        """
        self.model_path = model_path
        self.model = spacy.load(self.model_path)
        self.COLORS = {
            "TYPE": "#FFCCCC",
            "COLOR": "#CCFFCC",
            "STYLE": "#CCCCFF",
            "APPEARANCE": "#FFEECC",
            "ADDITIONAL_MATERIAL": "#DDCCFF",
            "FEATURE": "#FFFFCC"
        }
        self.doc = None

    def predict(self, text: str) -> List[Dict[str, Union[str, int, Dict[str, Union[str, int]]]]]:
        """
        Performs entity recognition on the given text and returns the detected entities.

        Args:
            text (str): The input text to perform entity recognition on.

        Returns:
            List[Dict[str, Union[str, int, Dict[str, Union[str, int]]]]]: A list of detected entities, each represented as a dictionary.

        Raises:
            None.
        """
        if self.doc is None or self.doc.text != text:
            self.doc = self.model(text)
        return self.doc.to_json()['ents']

    def display(self, text: str) -> None:
        """
        Displays the entity recognition visualization for the given text.

        Args:
            text (str): The input text to visualize.

        Returns:
            None.

        Raises:
            None.
        """
        if self.doc is None or self.doc.text != text:
            self.doc = self.model(text)
        return displacy.render(self.doc, style="ent", jupyter=True, options={'colors': self.COLORS})






