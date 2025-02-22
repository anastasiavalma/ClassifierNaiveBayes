from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer as NLPTokenizer, Stemmer, LemmatizerModel
import sparknlp
import time
from pyspark.sql.functions import concat_ws, trim, regexp_replace
import logging

spark = SparkSession.builder.appName("Reviews").config("spark.jars", "/home/ubuntu/ergasia3/spark-nlp_2.12-5.5.2.jar").config("spark.local.dir", "/storage").getOrCreate()

spark.sparkContext.setLogLevel("WARN")
logging.getLogger("py4j").setLevel(logging.WARN)

log_file_path = "results_log.txt"

# Φόρτωση δεδομένων από το HDFS
data_path = "hdfs://master:9000/user/ubuntu/input/Tools_and_Home_Improvement.jsonl"
data = spark.read.json(data_path)

# Γράψιμο αρχικών γραμμών του dataset
initial_row_count = data.count()
print(f"Initial rows: {initial_row_count}")


# Επιλέγουμε τις γραμμές που μας ενδιαφέρουν
data = data.select("rating", "text", "asin")

# Διαγραφή των γραμμών με rating = 3
data = data.filter((col("rating") != 3))

# Γράψιμο των γραμμώς μετά την αφαίρεση
rows_after_filtering_rating = data.count()
print(f"Rows after filtering out rating == 3: {rows_after_filtering_rating}")

# Καθαρίζουμε το text. Κανουμε trim σε αρχη, τελος και ενδιαμεσα του text
data = data.withColumn("text", trim(regexp_replace(col("text"), r"\s{2,}", " ")))

# Διαγράφουμε τα διπλότυπα βάσει του κειμένου και του asin
data = data.dropDuplicates(["text", "asin"])

# Γράψιμο των γραμμών μετά την διαγραφή των διπλοτύπων
rows_after_drop_duplicates = data.count()
print(f"Rows after removing duplicates: {rows_after_drop_duplicates}")

# Φτιάχνουμε τα labels. (1 για θετικό, 0 για αρνητικό)
data = data.withColumn("label", when(col("rating") >= 4, 1).otherwise(0))

# Διαγραφή των stop words
tokenizer = Tokenizer(inputCol="text", outputCol="words")
stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

# Stemming και lemmatization με το spark-nlp
document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

nlp_tokenizer = NLPTokenizer().setInputCols(["document"]).setOutputCol("token")

stemmer = Stemmer().setInputCols(["token"]).setOutputCol("stemmed")

lemmatizer = LemmatizerModel.pretrained().setInputCols(["token"]).setOutputCol("lemmatized")

finisher = Finisher().setInputCols(["lemmatized"]).setOutputCols("processed_text")

# Φτιάχνουμε το pipeline
nlp_pipeline = Pipeline(stages=[document_assembler, nlp_tokenizer, stemmer, lemmatizer, finisher])

# Εφαρμογή του pipeline
data = nlp_pipeline.fit(data).transform(data)

# Feature extraction
hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features")
idf = IDF(inputCol="raw_features", outputCol="features")
 
# Ορίζουμε το train / test split σύμφωνα με την εκφώνηση
train_test_splits = [(0.8, 0.2), (0.6, 0.4)]

with open(log_file_path, "w") as log_file:
    for train_frac, test_frac in train_test_splits:
        log_file.write(f"Train/Test split: {train_frac}/{1 - train_frac}\n")

        # Χωρισμός των δεδομένων
        train_data, test_data = data.randomSplit([train_frac, test_frac], seed=42)

        # Φτιάχνουμε το pipeline
        pipeline = Pipeline(stages=[tokenizer, stopwords_remover, hashing_tf, idf, NaiveBayes()])

        # Κρατάμε την αρχή για την μέτρηση
        start_time = time.time()

        # Εκπαίδευση του μοντέλου
        model = pipeline.fit(train_data)

        # Πρόβλεψη
        predictions = model.transform(test_data)

        # Κρατάμε το τέλος και βρίσουμε την διάρκεια
        end_time = time.time()
        total_time = end_time - start_time

        # Αποτελέσματα
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
        precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
        recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})

        # Γράφουμε τα αποτελέσματα
        log_file.write(f"Precision: {precision}, Recall: {recall}, Total Time: {total_time} seconds\n")
        print(f"Train/Test split: {train_frac}/{1 - train_frac}")
        print(f"Precision: {precision}, Recall: {recall}, Total Time: {total_time} seconds")

print("Finish!")
# Κλείνουμε το spark session
spark.stop()
