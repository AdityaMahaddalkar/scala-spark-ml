import ModelType.{DECISION_TREE_MODEL, LOGISTIC_REGRESSION_MODEL, ModelType, RANDOM_FOREST_MODEL}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.DataFrame

class ClassificationModel(modelType: ModelType) {

  private final val NUM_FOLDS: Int = 5

  private val model = modelType match {
    case LOGISTIC_REGRESSION_MODEL => new LogisticRegression()
    case RANDOM_FOREST_MODEL => new RandomForestClassifier()
    case DECISION_TREE_MODEL => new DecisionTreeClassifier()
  }


  def train_and_evaluate(df: DataFrame): Unit = {

    val cvModel = getCrossValidator.fit(df)

    println(cvModel.avgMetrics.mkString("Array(", ", ", ")"))
  }

  private def getCrossValidator: CrossValidator = {

    val paramGrid = new ParamGridBuilder()
      .build()

    val evaluator = new BinaryClassificationEvaluator()

    val crossValidator = new CrossValidator()
      .setEstimator(model)
      .setEstimatorParamMaps(paramGrid)
      .setEvaluator(evaluator)
      .setNumFolds(NUM_FOLDS)

    crossValidator
  }
}
