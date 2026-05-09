from deepeval.metrics import ContextualPrecisionMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# Define test case
test_case = LLMTestCase(
    input="What should I do if these shoes don't fit?",
    actual_output="We offer a 30-day no-questions-asked full refund.",
    expected_output="Customers can return items within 30 days and receive a full refund.",
    retrieval_context=["All customers are eligible for a 30-day no-questions-asked full refund."]
)

# Define evaluation metrics
contextual_precision = ContextualPrecisionMetric()
answer_relevancy = AnswerRelevancyMetric()

# Run evaluation
contextual_precision.measure(test_case)
answer_relevancy.measure(test_case)

print("Contextual precision score: ", contextual_precision.score)
print("Answer relevancy score: ", answer_relevancy.score)
