
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;


namespace SentimentAnalysis
{
    class Program
    {

        static readonly string s_dataPath = Path.Combine(Environment.CurrentDirectory, "..", "..", "..", "Data", "databool3.csv");
        private static string _modelPath => Path.Combine(Environment.CurrentDirectory, "..", "..", "..", "Models", "model.zip");


        static void Main(string[] args)
        {


            MLContext mlContext = new MLContext();


            if (!File.Exists(_modelPath))
            {


                TrainTestData splitDataView = LoadData(mlContext);



                ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);



                Evaluate(mlContext, model, splitDataView.TestSet);


                SaveModelAsFile(mlContext, splitDataView.TrainSet.Schema, model);


                UseModelWithSingleItem(mlContext, model);

            }

            UseModelWithBatchItems(mlContext);



            Console.WriteLine();
            Console.WriteLine("=============== End of process ===============");
        }

        public static TrainTestData LoadData(MLContext mlContext)
        {
            // Загрузка данных их файла
            // Метод LoadFromTextFile() определяет схему данных и считывает файл.
            // Он принимает переменные, содержащие пути к данным, и возвращает объект IDataView.

            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(s_dataPath, hasHeader: true, separatorChar: ';', allowQuoting: true);


            // Метод TrainTestSplit() позволяет разделить загруженный набор данных на учебный  
            // и проверочный наборы данных и возвратить их в класс DataOperationsCatalog.TrainTestData.
            // Процент тестовых данных указывается с помощью параметра testFraction. Значение по умолчанию — 10 %

            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);


            return splitDataView;

        }

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            // Метод FeaturizeText()  преобразует текстовый столбец (SentimentText) в числовой столбец типа ключа Features.

            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))

            // SdcaLogisticRegressionBinaryTrainer —  алгоритм обучения классификации.
            // Он добавляется в estimator и принимает SentimentText с присвоенными признаками (Features)
            // и входные параметры Label, чтобы пройти обучение по  данным.

            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));


            // Метод Fit() обучает модель путем преобразования набора данных и применения обучения.

            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(splitTrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            return model;

        }

        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            // После обучения модели с помощью проверочных данных производится валидация производительности модели.

            // Метод Transform() делает прогнозы для нескольких входных строк тестового набора данных.

            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);


            // После получения прогноза (predictions) метод Evaluate() оценивает модель, 
            // сравнивая спрогнозированные значения с фактическими метками (Labels) в тестовом наборе данных,
            // а затем возвращает объект CalibratedBinaryClassificationMetrics как метрики эффективности модели.    

            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");


            // Метрика Accuracy возвращает точность модели — это доля правильных прогнозов в тестовом наборе.

            // Метрика AreaUnderRocCurve показывает уверенность модели в правильности классификации.
            // Значение AreaUnderRocCurve должно быть максимально близким к единице.

            // Метрика F1Score содержит F1-оценку модели, который является мерой баланса между точностью и полнотой.
            // Значение F1Score должно быть максимально близким к единице.


            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");


        }

        private static void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {

            mlContext.Model.Save(model, trainingDataViewSchema, _modelPath);


            Console.WriteLine("The model is saved to {0}", _modelPath);
        }

        // Метод UseModelWithSingleItem() выполняет следующие задачи:
        // прогноз тональности на основе тестовых данных, отображение результатов прогнозирования

        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {

            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);



            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = "This was a very bad steak"
            };


            var resultPrediction = predictionFunction.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();

        }

        public static string GetPrediction(double probability)
        {
            if (probability > 0.6) return "Positive";
            if (probability < 0.4) return "Negative";
            return "Neutral";
        }

        public static void UseModelWithBatchItems(MLContext mlContext)
        {
            ITransformer model = mlContext.Model.Load(_modelPath, out DataViewSchema modelInputSchema);

            string[] text = new[]
            {
                "очень плохо ужасно ненависть неудача грустно ",
                "СК начал проверку из-за открытия пассажиром аварийного люка самолёта",
                "В центре Курска часть домов осталась без горячей воды",
                "В Курске в ДТП перевернулась ГАЗель",
                "Маск и Безос поздравили Брэнсона с успешным полетом к границе космоса",
                "Настолько неприятных людей я еще не видела",
                "И это почти 10 лет спустя... грустная песенка",
                "Не переношу зубную боль,а к стоматологу даже нет времени записаться",
                "что то жестко лагает по несколько секунд. играть невозможно. танк разбирают за это время(",
                " ненавижу скриншоты и фото в обзоры добавлять",
                "Всегда бы такую погоду как сегодня. мороза нет, ветра нет и только снег падает) красота",
                "Смотрится гораздо красивее купленных нами БКМов... Не правда ли",
                "Смотрится гораздо красивее купленных нами БКМов... ",
                "И сон сегодня ПРИКОЛЬНЫЙ приснился;) , а то вечно кошмарики. ;)",
                "Такая искренняя радость) они заслужили эту победу)))",
                "учебная программа по ритмике для дши",
                "В Японии приняли закон о хранении тайн",
                "сдам комнату в краснодаре в районе схи",
                "В любой непонятной ситуации чисти кэш.",
                "как поэтапно нарисовать кувшин гуашью",
                "маршрут от дома до школы для портфолио",
                "инструкция к применению зеленого кофе",
                "поручику подходит полковник и говорит",
                "сколько каллорий употреблять на сушке",
                "программа для создания проектов кухни",
                "шпаргалки для олимпиады по математике",
                "как правильно заваривать зеленый кофе",
                "поделки из пластиковых бутылок пальма",
                "интерактивный семинар по общей физике",
                "сочинение на тему в гости к египтянину"
            };


            IEnumerable<SentimentData> sentiments = Enumerable.Empty<SentimentData>();

            foreach (string s in text)
            {
                SentimentData sd = new SentimentData
                {
                    SentimentText = s
                };
                sentiments = sentiments.Append(sd);
            }



            // Использование модели для прогнозирования тональности комментариев с помощью метода Transform()

            IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

            IDataView predictions = model.Transform(batchComments);

            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);



            Console.WriteLine();

            Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");


            Console.WriteLine();


            foreach (SentimentPrediction prediction in predictedResults)
            {
                Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {GetPrediction(prediction.Probability)} | Probability: {prediction.Probability} ");
            }

            Console.WriteLine("=============== End of predictions ===============");

        }
    }
}
