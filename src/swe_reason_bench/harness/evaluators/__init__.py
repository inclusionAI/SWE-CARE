from abc import ABC, abstractmethod
from typing import Any, Optional

from loguru import logger


class Evaluator(ABC):
    """Grade, tag, or otherwise evaluate predictions relative to their inputs
    and/or reference labels."""

    @property
    def evaluation_name(self) -> str:
        """The name of the evaluation."""
        return self.__class__.__name__

    @property
    def requires_reference(self) -> bool:
        """Whether this evaluator requires a reference label."""
        return False

    @property
    def requires_input(self) -> bool:
        """Whether this evaluator requires an input."""
        return False

    @property
    def _skip_input_warning(self) -> str:
        """Warning to show when input is ignored."""
        return f"Ignoring input in {self.__class__.__name__}, as it is not expected."

    @property
    def _skip_reference_warning(self) -> str:
        """Warning to show when reference is ignored."""
        return (
            f"Ignoring reference in {self.__class__.__name__}, as it is not expected."
        )

    def _check_evaluation_args(
        self,
        reference: Optional[Any] = None,
        input: Optional[Any] = None,
    ) -> None:
        """Check if the evaluation arguments are valid.

        Args:
            reference (Optional[Any], optional): The reference label.
            input (Optional[Any], optional): The input.
        Raises:
            ValueError: If the evaluator requires an input but none is provided,
                or if the evaluator requires a reference label but none is provided.
        """
        if self.requires_input and input is None:
            raise ValueError(f"{self.__class__.__name__} requires an input.")
        elif input is not None and not self.requires_input:
            logger.warning(self._skip_input_warning)
        if self.requires_reference and reference is None:
            raise ValueError(f"{self.__class__.__name__} requires a reference.")
        elif reference is not None and not self.requires_reference:
            logger.warning(self._skip_reference_warning)

    @abstractmethod
    def _evaluate(
        self,
        *,
        prediction: Any,
        reference: Optional[Any] = None,
        input: Optional[Any] = None,
        **kwargs: Any,
    ) -> dict:
        """Evaluate Chain or LLM output, based on optional input and label.

        Args:
            prediction (Any): The LLM or chain prediction to evaluate.
            reference (Optional[Any], optional): The reference label to evaluate against.
            input (Optional[Any], optional): The input to consider during evaluation.
            kwargs: Additional keyword arguments, including callbacks, tags, etc.
        Returns:
            dict: The evaluation results containing the score or value.
                It is recommended that the dictionary contain the following keys:
                     - score: the score of the evaluation, if applicable.
                     - value: the string value of the evaluation, if applicable.
                     - reasoning: the reasoning for the evaluation, if applicable.
        """  # noqa: E501

    def evaluate(
        self,
        *,
        prediction: Any,
        reference: Optional[Any] = None,
        input: Optional[Any] = None,
        **kwargs: Any,
    ) -> dict:
        """Evaluate Chain or LLM output, based on optional input and label.

        Args:
            prediction (Any): The LLM or chain prediction to evaluate.
            reference (Optional[Any], optional): The reference label to evaluate against.
            input (Optional[Any], optional): The input to consider during evaluation.
            kwargs: Additional keyword arguments, including callbacks, tags, etc.
        Returns:
            dict: The evaluation results containing the score or value.
        """  # noqa: E501
        self._check_evaluation_args(reference=reference, input=input)
        return self._evaluate(
            prediction=prediction, reference=reference, input=input, **kwargs
        )
