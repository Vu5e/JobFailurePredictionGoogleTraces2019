# Job Failure Prediciton In Cloud With Imbalanced Dataset Handling Techniques
By learning and using prediction for failures, it is one of the important steps to improve the reliability of the cloud computing system. Furthermore, gave the ability to avoid incidents of failure and costs overhead of the system. It created a wonderful opportunity with the breakthroughs of machine learning and cloud storage that utilize generated huge data that provide pathways to predict when the system or hardware malfunction or fails. It can be used to improve the reliability of the system with the help of insights of using statistical analysis on the workload data from the cloud providers. This research will discuss regarding job usage data of tasks on the large “Google Cluster Workload Traces 2019” dataset, using multiple resampling techniques such as “Random Under Sampling, Random Oversampling and Synthetic Minority Oversampling Technique” to handle the imbalanced dataset. Furthermore, using multiple machine learning algorithm which is for traditional machine learning algorithm are “Logistic Regression, Decision Tree Classifier, Random Forest Classifier, Gradient Boosting Classifier and Extreme Gradient Boosting Classifier” while deep learning algorithm using “Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU)” for job failure prediction between imbalanced and balanced dataset. Then, to have a comparison of imbalanced and balanced in terms of model accuracy, error rate, sensitivity, f – measure, and precision. The results are Extreme Gradient Boosting Classifier and Gradient Boosting Classifier is the most performing algorithm with and without imbalanced handling techniques. It showcases that SMOTE is the best method to choose from for handling imbalanced data. The deep learning model of LSTM and Gated Recurrent Unit may be not the best for the in terms of accuracy, based on the ROC Curve its better than the XGBoost Classifier and Gradient Boosting Classifier.

## Introduction
![Proposed Method](https://github.com/Vu5e/JobFailurePredictionGoogleTraces2019/blob/main/Raw/Proposed%20Method/Picture1.png)

A brief introduction on the theoretical framework of this study will be provided on this section and will be presenting the imbalance classifier methods used and machine learning techniques that will be used to design the failure prediction models. The conceptual map of research methodology in this paper will simplified in Figure 3.1. 

## Data Understanding

By analysing the raw data, important features and information can be found as the importance of data understanding. The scope of data understanding in this study are mainly the job and task in the Google Cluster Workload Traces Dataset 2019 which contribute to the job and task failures which to understand below:
 - The 8-cell running and describing usage traces from a single Borg cell.
 -	The CollectionEvents table and InstanceEvents table describe the life cycle of collections and instances, respectively.
 -	The Resource usage using Linux containers for resources isolation and usage accounting.

First is to understand, there are 8 cell running several days and it describe usage traces for several days from a single Borg cell. The trace of jobs and task are described below: 
 - Several tables are made up in a trace, the primary key is each indexed that usually includes a timestamp.
 - Individual machines and information are provided by the cell management system which makes it a table for the data.
 - Timestamp in each record, which before the beginning is in microseconds in the past “600 seconds” and recorded as “64 – bit integer” of the trace period, the unique 64 – bit identifier is assigned to every job and machine.
 - The job ID is typically identified in the task and for jobs it used a 0 – based index. 
 - The workload consists of six (6) separate tables which are job events, task events, machine events, machine attributes, machine constraints and alloc set. 

Next, are the CollectionEvents table and InstanceEvents table which portray the life cycle of collections and instances. The CollectionEvents and InstanceEvents are mainly features are used to know which feature that contribute to the job failure. The description regarding the CollectionEvents and InstanceEvents are described below:
 - All the tasks within a job usually execute the same binary with the same options and resource request.
 - A common pattern is to run masters (controllers) and workers in separate jobs (e.g., this is used in MapReduce and similar systems).
 - A worker job that has a master job as a parent will automatically be terminated when the parent exits, even if its workers are still running and a job can have multiple child jobs but only one parent.
 - Another pattern is to run jobs in a pipeline, if job A says that it should run after job B, then job A will only be scheduled (made READY) after job B successfully finishes.
 - A job can list multiple jobs that it should run after and will only be scheduled after all those jobs finish successfully.

Finally, the Resource usage using Linux containers for resources isolation and usage accounting. The resource usage will be mainly used to calculate the power consumption of the usage in each cell, which will be explained below in the section 3.8 Power Consumption. Each task runs within its own container and may create multiple processes in that container. 
 - Alloc instances are also associated with a container, inside which task containers nest. 
 - The report usage values from a series of non – overlapping measurement windows for each instance. 
 - The windows are typically 5 minutes long, although may be shorter if the instance starts, stops, or is updated within that period. 
 - The measurements may extend for up to tens of seconds after an instance is terminated, or (rarely) for a few minutes.

### Downloading the Dataset
The sources of data are retrieved from Google Cloud Storage. The total size of the compress trace is approximately 2.4TB. Since the dataset is big a sample of the dataset will be chosen and the method to process the data will be using Jupyter Notebook + Dask. Further explanation will be explained in section 3.3 Data Preparation. The dataset can be downloaded directly from Google Cloud Platform but by using HTTP, the google library which is known as urllib. request for an extensible library to open the URLs link used to download the files from the common storage. The urllib. request for modules which define functions and classes to facilitate the opening of URLs. It can be either a string or a request object. Figure 3.2 shows the process of extracting the data directly from HTTP using urllib. request. The information about the dataset is available on the website https://github.com/google/cluster-data/blob/master/ClusterData2019.md. Connected by high bandwidth cluster network, packed with and into physical racks, it’s what set of machines on trace is. 

### Data Extraction
The whole dataset is a total size of “133 GB” compressed size. Extracting the json file from files which have been downloaded from the google website by using HTTP. New data folder has been created to store the data which has been download. Downloaded data from the common storage need to be converted into proper form. Then, from the downloaded files, using “7Zip” to extract the json file for job and task files from common storage (.gz). After that by using the schemas provided, it’ll easy to query required files out of five schemas provided by Google Cloud.

## Data Preparation
Once the files have been downloaded and extracted, the data is ready to be downloaded to the local storage for the modification process. The reason to use PowerBI is because of the large files of the traces dataset that is been handled. We can use python but since the dataset is in JSON format, it’ll take a long time to convert it to CSV file format and mostly causes crashes to the workstation used for this study. The solution found is by using PowerBI which can handle the large dataset provided by the Google Traces.

The “collection_events”, “instance_events”. This process will also be done for “instance_usage”. The timestamp on each dataset is in character form, hence, to analyse the data, the timestamp needs to be converted to become numeric. The converted timestamp will turn to microsecond number, and the purpose of converting timestamp from ‘char’ data type to microsecond number is because it’ll useful to calculate and visualize the failure based on numeric timestamp instead of using the character forms. The figure 3.3 shows the overall look and features contained using PowerBi for the Google Cluster Workload Traces Dataset 2019.

As “PowerBI” has limitations for exporting the file into CSV for rows that exceed 150,000, using “Dax Studio” can overcome these limitations. The rows that the dataset has currently is 2,065,730 and with “Dax Studio” this can be easily exported as CSV file type. Figure 3.4 shows the process of exporting the dataset in “Dax Studio”.

### Dataset Append
![Append Dataset Method](https://github.com/Vu5e/JobFailurePredictionGoogleTraces2019/blob/main/Raw/Proposed%20Method/Picture2.png)

After changing the file type to CSV, based on Figure 3.5, first the dataset needs to be appended based on the collection_events, instance_usage, and instance_events. The dataset is appended first because of the large size in Google Cluster Workload Traces Dataset 2019. By using Dask + Jupyter Notebook, large dataset can be handled easily using lazy computation. Figure below shows the code to append, compute the value of each feature and outputting the file as CSV after the file is appended.

The collection_events are read first as it is separated to partitions. After that in Figure 3.7, the collection_events will be appended using concat method. In Python, the append function is used to add a single item to the list. A new list of items is not returned, but the existing list is modified by adding the new item to its end. The dataset in Figure 3.6 and Figure 3.7 are “collection_events” A, we can start on computing the number of rows it has on the dataset. Figure below showcase the “collection_events” A dataset. All of this are done for the “instance_events” and “instance_usage” as well from Cell A to H. Finally, after the dataset has been append it will save into an appended CSV dataset as a single file. Figure below showcase the appended dataset saved as CSV.

### Dataset Merge
![Merge Dataset Method](https://github.com/Vu5e/JobFailurePredictionGoogleTraces2019/blob/main/Raw/Proposed%20Method/Picture3.png)

After appending the dataset, based on Figure, the dataset needed to be merge using the method shown in Figure 3.10. Figure 3.11 below shows the code for merging dataset. "Merging" two datasets is the act of putting two datasets together into one and aligning the rows from each based on similar properties or columns. Other languages, as well as Pandas, employ "merge" and "join" in a similar manner. Both the "merge" and the "join" functions in Pandas perform the same thing in practise.

The “Collection_Events”, “Instance_Usage”, and “Instance_Events” are merged, filtered by the type and resource column. The filter is based on 4: Evict, 5: Fail, 7: Kill and 8: Lost as failed jobs = 0. Successful jobs = 1 is filtered for number 6: Finish. All of this are done using Jupyter Notebook + Dask with the help of schemas provided by Google Cloud. Figure 3.12 shows the code to filter the “type” feature for failed and successful jobs.

### Data Cleaning
After the dataset has been merged and filtered, data are cleaned from null values and removing features that doesn’t correlate with feature “type”. Figure below showcase the correlation to the features "type".

![Correlation of the Feature to the "type"](https://github.com/Vu5e/JobFailurePredictionGoogleTraces2019/blob/main/Raw/Proposed%20Method/Picture4.png)

The features are removed based on the threshold of negative value below 0 but there are certain required features that is needed such as cpu_usage_distribution to be used for power consumption formula and will be kept. The null values found are filled using interpolation backfill method. Finally, the merged job is saved as a single CSV file using Dask Library.

### Data Reduction
Since there are limitations to the workstation used in this study, the dataset is reduced to 1 million of rows. The dataset is kept from the first top of 0 to 999999 of row and the remaining bottom is removed. Below Figure 3.17 shows the dataset the computation of the dataset after data reduction.

![Data Reduction](https://github.com/Vu5e/JobFailurePredictionGoogleTraces2019/blob/main/Raw/Proposed%20Method/Picture5.png)

### Data Transformation
![Imbalanced Data Found on the "type" features](https://github.com/Vu5e/JobFailurePredictionGoogleTraces2019/blob/main/Raw/Proposed%20Method/Picture6.png)

Based on the Figure above this the number of failed jobs = 0 and successful jobs = 1. Since the dataset is highly imbalanced, the imbalance classifier is applied to it. Before handling the imbalanced dataset, we need to explore whether if there’s any outlier on the data. Based on the Figure 3.19 the dataset is mostly right skewed on feature “resource_request.cpus”, “resource_request.memory” and “memory_access_per_instruction”. Since this is a time series dataset, a Log Transformer Power from “sklearn” library are applied towards it. By using Log Power Transformer Power, it can make the data have a normal distribution so that statistical analysis results can be done, and the data is more valid now. To make data more Gaussian-like, power transforms are a type of parametric, monotonic transformation. Useful in cases when normalcy is sought, such as simulating heteroscedasticity (non-constant variance). Based on the dataset it doesn’t reduce the dataset values and it uses Yeo-Johnson which supports both positive and negative data. Here are the results at Figure 3.20 after handling the skewness of the data. After that the imbalanced calassifier will be implemented based on the figure below.

![Data Sampling](https://github.com/Vu5e/JobFailurePredictionGoogleTraces2019/blob/main/Raw/Proposed%20Method/Picture7.png)

As shown in Figure, the dataset is quite imbalanced as successful job is more than the failed jobs in the “type” column. So before modelling, the data is going through imbalance classifier to solve the imbalanced in the dataset. There are three (3) imbalance classifier that are chosen to handle the imbalanced dataset. Which is “Random Under sampling (RUS)”, “Random Oversampling (ROS)” and “Synthetic Minority Oversampling Technique (SMOTE)”.

### Power Consumption
Based on the given features that available in the dataset, it is possible to add a synthetic column that contain the power consumption given each row. The formula for the power consumption is as below: 

- P(x_t)=P(0%)+(P(100%)-P(0%))(2x_t-x_t^1.4)

Based on the formula given above, x_t denotes of CPU utilization of the server at given time t. P(100%) denotes the power consumption of the server in full load, and P(0%) denotes the power consumption of the server in the idle mode. Based on the paper, the power consumption at P(100%) which is at full load set at 145W and P(0%) is set at 87W (Liu et al., 2017). Selected features to produce the power consumption are “cpu_usage”. Below Figure 3.22 shows the code to run the power consumption formula.

### Data Storage
After data transformation, as shown in the Figure, the dataset is stored and save in CSV file format ready to be used for predictive modelling.

## Data Modelling

Data modelling will take place after the data storage process, the machine learning algorithm is applied on the data to create a prediction model. Before using the machine learning for failure prediction, the data is going through a cross validation first with a simple split of 70% and 30% for training and validating the dataset, respectively.

### Traditional Machine Learning Algorithm
After that, there are five (5) machine learning algorithms which used to create the failure prediction model for Google Cloud trace dataset which are describe below:
•	Logistic Regression
•	Decision Tree Classifier
•	Random Forest Classifier
•	Gradient Boosting Classifier
•	Extreme Gradient Boosting Classifier

### Deep Learning Algorithm
Deep Learning techniques will also be used in this study using tensorflow and keras, which is describe in table below:
•	Long Short-Term Memory (LSTM)
•	Gated Recurrent Unit (GRU)

This deep learning model are running with 100 epochs. This method is using with help of “Jupyter Notebook + Dask + imbalanced – learn + scikit + sklearn + keras + tensorflow”. As the dataset is quite big with 1 million plus row “Dask” can be used to help process the data faster. “Dask” enable efficient parallel computations on single machines by leveraging their multi-core CPUs and streaming data efficiently from disk. “Imbalanced – learn” library is used to run the imbalanced classifier chosen while “scikit” and “sklearn” is used to run the chosen data modelling as explained above.

## Model Evaluation
Model evaluation is one of the most important parts in validating the quality of the model by comparing its performance with other algorithm used to create the best prediction model to predict failure prediction in Google Cloud service. 

### Evaluation Metrics
The performance of each model is to compare based on model accuracy, error rate, sensitivity, precision, and F – measure. 
Below are formulas for the model evaluation in terms of model accuracy, error rate sensitivity, precision, and f – measure. First section of the formula describes regarding the “True Positives (TP), True Negatives (TN), False Positives (FP) and False Negatives (FN) followed by model accuracy, error rate, sensitivity, precision and f – measure as follows: 
-	True Positives (TP) – Correctly predicted positive values of the class actual value is yes, predicted value of class will also be yes. 
-	True Negatives (TN) – Correctly predicted positive values of the class actual value is no, predicted value of class will also be no 
-	False Positives (FP) – When the predicted class is yes but the actual value class is no. 
-	False Negatives (FN) – When the predicted class is yes but actual value class in no. 

### Job Level Failure Predicition
In this section as well is an addition of explainable AI using Dalex, this package is and explainable artificial intelligence that can analyse and x-ray models to explore and explain its behaviour. This will help on understanding the complex models that are been working on (Baniecki et al., 2021)

Based on Figure, Dalex main function called explain () creates a wrapper around the predictive model. Then based on the wrapped models, explanation and comparison are brought up with collection of global and local explainers. Using Dalex as an explainable AI or tools can help gained insight on how the model run for the job failure prediction of Google Trace 2019 dataset (Baniecki et al., 2021). Using Dalex functions, features that are used are Variable Importance, Performance Access and ROC Curve.

![Variable Importance](https://github.com/Vu5e/JobFailurePredictionGoogleTraces2019/blob/main/Raw/Proposed%20Method/Picture11.png)

![ROC Curve](https://github.com/Vu5e/JobFailurePredictionGoogleTraces2019/blob/main/Raw/Proposed%20Method/Picture10.png)

## Result and Findings
This study aims to ascertain which features that bring impacts to the job failures in Google Cluster Workload Traces Dataset. Based on the Google Cluster Workload Dataset the features were produced and published in year 2019. According to the findings the probable cause that determine the job failure in the Google Cloud Workload cloud servers are the CPU usage, assigned memory and RAM request. Usually, each time there’s a limit to usage of CPU set by the cloud provider and users that demand more power for their CPU usage while having lesser power would trigger the job failure in the cloud system. While assigned memory and RAM request for that user set which has lesser RAM, than what they have insisted, which caused job failure in the cloud system. Important features in the findings of this study are which can predict job failures accurately are the collection id, priority to complete the task and jobs, CPU request, collection type, scheduler, and the power consumption.

This study offers an efficacious methodology to predict job failure in Google Cluster Workload Dataset by using logistic regression, decision tree, random forest classifier, gradient boosting, XGBoost, LSTM and Gated Recurrent Unit. Since the dataset is imbalanced, the comparison of using multiple imbalanced handling classifier also been introduced which is Random Undersampling, Random Oversampling and SMOTE. XGBoost Classifier model using imbalanced classifier using SMOTE shows the highest F-measure and accuracy during training and testing. The job failure has successfully predicted the failure using XGBoost Classifier with a sensitivity value of more than 90%. The F – measures for XGBoost Classifier shows the highest value at 0.99302 and it’s considered the best algorithm among other models. This demonstrates that the initial goal of determining how the strategies of unbalanced dataset handling might potentially improve the performance of failure prediction models has been met. A failure prediction model with and without unbalanced dataset handling was successfully modelled and developed, as was the second goal.

Power consumption which is a new column that has been added by using CPU usage plus formula derived from this study (Liu et al., 2017). From the result of variable importance, power consumption does contribute to the job failure of the Google Cluster Workload Traces Dataset 2019.

Finally, the goal of evaluating the classifier's performance using explainable AI methodologies was met, based on the ROC Curve presented in the figures using Dalex, the best smoothing curve with better positive rate is by using the SMOTE imbalanced classifier. Even though the LSTM and Gated Recurrent Unit is not the best or highest in accuracy for this study, the ROC Curve shows that the Gated Recurrent Unit and LSTM has the best consistent and ROC curve with and without imbalanced handling. Thus, LSTM and Gated Recurrent Unit, can still be used to be one of the best performing algorithms beside XGBoost Classifier in predicting job failure in the cloud system.  Add on is that the deep learning needs to be add with more complex parameters and hidden layers to further increased the accuracy and reduce the loss or error to further improved the accuracy. Overall, it can be concluded that the experiment conducted by using and comparing the seven-machine learning algorithm with imbalanced handling classifier has successfully reached the objective of this study. 

![ROC Curve Results](https://github.com/Vu5e/JobFailurePredictionGoogleTraces2019/blob/main/Raw/Results/Picture13.png)

# References
Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., Devin, M., Ghemawat, S., Irving, G., Isard, M., Kudlur, M., Levenberg, J., Monga, R., Moore, S., Murray, D. G., Steiner, B., Tucker, P., Vasudevan, V., Warden, P., … Zheng, X. (2016). TensorFlow: A system for large-scale machine learning. Proceedings of the 12th USENIX Symposium on Operating Systems Design and Implementation, OSDI 2016. https://doi.org/10.5555/3026877.3026899

Abdul, A., Vermeulen, J., Wang, D., Lim, B. Y., & Kankanhalli, M. (2018). Trends and trajectories for explainable, accountable and intelligible systems: An HCI research agenda. Conference on Human Factors in Computing Systems - Proceedings, 2018-April. https://doi.org/10.1145/3173574.3174156

Abubakar, A., Barbhuiya, S., Kilpatrick, P., Vien, N. A., & Nikolopoulo, D. S. (2020). Fast analysis and prediction in large scale virtual machines resource utilisation. CLOSER 2020 - Proceedings of the 10th International Conference on Cloud Computing and Services Science. https://doi.org/10.5220/0009408701150126

Agarwal, H., & Sharma, A. (2016). A comprehensive survey of Fault Tolerance techniques in Cloud Computing. 2015 International Conference on Computing and Network Communications, CoCoNet 2015. https://doi.org/10.1109/CoCoNet.2015.7411218

Al-Dhuraibi, Y., Paraiso, F., Djarallah, N., & Merle, P. (2018). Elasticity in Cloud Computing: State of the Art and Research Challenges. IEEE Transactions on Services Computing, 11(2). https://doi.org/10.1109/TSC.2017.2711009

Al-Raheym, S., Açan, S. C., & Pusatli, Ö. T. (2016). Investigation Of Amazon And Google For Fault Tolerance Strategies In Cloud Computing Services. AJIT-e Online Academic Journal of Information Technology, 7(23). https://doi.org/10.5824/1309-1581.2016.4.001.x

Alam, T. M., Shaukat, K., Hameed, I. A., Luo, S., Sarwar, M. U., Shabbir, S., Li, J., & Khushi, M. (2020). An investigation of credit card default prediction in the imbalanced datasets. IEEE Access, 8. https://doi.org/10.1109/ACCESS.2020.3033784

Alber, M., Lapuschkin, S., Seegerer, P., Hägele, M., Schütt, K. T., Montavon, G., Samek, W., Müller, K. R., Dähne, S., & Kindermans, P. J. (2019). INNvestigate neural networks! Journal of Machine Learning Research, 20.

Anaconda Inc. (2021). Anaconda. Https://Www.Anaconda.Com/.

Angarita, R., Rukoz, M., Manouvrier, M., & Cardinale, Y. (2016). A knowledge-based approach for self-healing service-oriented applications. 8th International Conference on Management of Digital EcoSystems, MEDES 2016. https://doi.org/10.1145/3012071.3012100

Arya, V., Bellamy, R. K. E., Chen, P. Y., Dhurandhar, A., Hind, M., Hoffman, S. C., Houde, S., Liao, Q. V., Luss, R., Mojsilović, A., Mourad, S., Pedemonte, P., Raghavendra, R., 

Richards, J. T., Sattigeri, P., Shanmugam, K., Singh, M., Varshney, K. R., Wei, D., & Zhang, Y. (2020). Ai explainability 360: An extensible toolkit for understanding data and machine learning models. Journal of Machine Learning Research, 21.

Baniecki, H., Kretowicz, W., Piatyszek, P., Wisniewski, J., & Biecek, P. (2021). dalex: Responsible machine learning with interactive explainability and fairness in python. Journal of Machine Learning Research, 22.

Barredo Arrieta, A., Díaz-Rodríguez, N., Del Ser, J., Bennetot, A., Tabik, S., Barbado, A., Garcia, S., Gil-Lopez, S., Molina, D., Benjamins, R., Chatila, R., & Herrera, F. (2020). Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI. Information Fusion, 58. https://doi.org/10.1016/j.inffus.2019.12.012

Barroso, L. A., Clidaras, J., & Hölzle, U. (2013). The datacenter as a computer: An introduction to the design of warehouse-scale machines, second edition. Synthesis Lectures on Computer Architecture, 24. https://doi.org/10.2200/S00516ED2V01Y201306CAC024

Bellamy, R. K. E., Dey, K., Hind, M., Hoffman, S. C., Houde, S., Kannan, K., Lohia, P., Martino, J., Mehta, S., Mojsilovic, A., Nagar, S., Ramamurthy, K. N., Richards, J., Saha, D., Sattigeri, P., Singh, M., Varshney, K. R., & Zhang, Y. (2018). Ai fairness 360: An extensible toolkit for detecting, understanding, and mitigating unwanted algorithmic bias. ArXiv.

Charles Reiss, John Wilkes, & Joseph Hellerstein. (2011). Google cluster-usage traces: format + schema. Google.

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16. https://doi.org/10.1613/jair.953

Chawla, N. V, Japkowicz, N., & Drive, P. (2004). Editorial : Special Issue on Learning from Imbalanced Data Sets Aleksander Ko l cz. ACM SIGKDD Explorations Newsletter, 6(1).

Chen, G., Jin, H., Zou, D., Zhou, B. B., Qiang, W., & Hu, G. (2010). SHelp: Automatic self-healing for multiple application instances in a virtual machine environment. Proceedings - IEEE International Conference on Cluster Computing, ICCC. https://doi.org/10.1109/CLUSTER.2010.18

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 13-17-August-2016. https://doi.org/10.1145/2939672.2939785

Chen, X., Lu, C. Da, & Pattabiraman, K. (2014a). Failure analysis of jobs in compute clouds: A google cluster case study. Proceedings - International Symposium on Software Reliability Engineering, ISSRE. https://doi.org/10.1109/ISSRE.2014.34

Chen, X., Lu, C. Da, & Pattabiraman, K. (2014b). Failure prediction of jobs in compute clouds: A Google cluster case study. Proceedings - IEEE 25th International Symposium on Software Reliability Engineering Workshops, ISSREW 2014. https://doi.org/10.1109/ISSREW.2014.105

Cheng, Y., Chai, Z., & Anwar, A. (2018). Characterizing co-located datacenter workloads: An alibaba case study. Proceedings of the 9th Asia-Pacific Workshop on Systems, APSys 2018. https://doi.org/10.1145/3265723.3265742

Chris Parmer, & Nicolas Kruchten. (2020). plotly: An open-source, interactive data visualization library for Python. GitHub.

David Cournapeau. (2007). Scikit-learn. Https://Scikit-Learn.Org/Stable/Index.Html.

Dinu, F., & Ng, T. S. E. (2012). Understanding the effects and implications of compute node related failures in Hadoop. HPDC ’12 - Proceedings of the 21st ACM Symposium on High-
Performance Parallel and Distributed Computing. https://doi.org/10.1145/2287076.2287108

Dubey, K., Shams, M. Y., Sharma, S. C., Alarifi, A., Amoon, M., & Nasr, A. A. (2019). A Management System for Servicing Multi-Organizations on Community Cloud Model in Secure Cloud Environment. IEEE Access, 7. https://doi.org/10.1109/ACCESS.2019.2950110

Egwutuoha, I. P., Chen, S., Levy, D., & Selic, B. (2012). A fault tolerance framework for high performance computing in cloud. Proceedings - 12th IEEE/ACM International Symposium on Cluster, Cloud and Grid Computing, CCGrid 2012. https://doi.org/10.1109/CCGrid.2012.80

El-Sayed, N., Zhu, H., & Schroeder, B. (2017). Learning from Failure Across Multiple Clusters: A Trace-Driven Approach to Understanding, Predicting, and Mitigating Job Terminations. Proceedings - International Conference on Distributed Computing Systems. https://doi.org/10.1109/ICDCS.2017.317

Eli Cortez, Microsoft GitHub User, Microsoft Open Source, & Rodrigo Fonseca. (2017). Microsoft Azure Traces. GitHub.
Elzamly, A., & Hussin, B. (2016). Classification of critical cloud computing security issues for banking organizations: A cloud delphi study. International Journal of Grid and Distributed Computing, 9(8). https://doi.org/10.14257/ijgdc.2016.9.8.13

Fadishei, H., Saadatfar, H., & Deldari, H. (2009). Job failure prediction in grid environment based on workload characteristics. 2009 14th International CSI Computer Conference, CSICC 2009. https://doi.org/10.1109/CSICC.2009.5349381

Feldman, M., Friedler, S. A., Moeller, J., Scheidegger, C., & Venkatasubramanian, S. (2015). Certifying and removing disparate impact. Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2015-August. https://doi.org/10.1145/2783258.2783311

Feng, Y., Wang, T., Hu, B., Yang, C., & Tan, J. (2020). An integrated method for high-dimensional imbalanced assembly quality prediction supported by edge computing. IEEE Access, 8. https://doi.org/10.1109/ACCESS.2020.2988118

Figone, S. (2020). Scaling in the Cloud. RapidScale.

François Chollet. (2015). Keras. Https://Keras.Io/Api/.

Galar, M., Fernandez, A., Barrenechea, E., Bustince, H., & Herrera, F. (2012). A review on ensembles for the class imbalance problem: Bagging-, boosting-, and hybrid-based approaches. In IEEE Transactions on Systems, Man and Cybernetics Part C: Applications and Reviews (Vol. 42, Issue 4). https://doi.org/10.1109/TSMCC.2011.2161285
Gao, J., Wang, H., & Shen, H. (2020). Task Failure Prediction in Cloud Data Centers Using Deep Learning. IEEE Transactions on Services Computing. https://doi.org/10.1109/TSC.2020.2993728

Garg, A., & Bagga, S. (2016). An autonomic approach for fault tolerance using scaling, replication and monitoring in cloud computing. Proceedings of the 2015 IEEE 3rd International Conference on MOOCs, Innovation and Technology in Education, MITE 2015. https://doi.org/10.1109/MITE.2015.7375302

Garraghan, P., Townend, P., & Xu, J. (2014). An empirical failure-analysis of a large-scale cloud computing environment. Proceedings - 2014 IEEE 15th International Symposium on High-Assurance Systems Engineering, HASE 2014. https://doi.org/10.1109/HASE.2014.24

Gill, S. S., & Buyya, R. (2018). A taxonomy and future directions for sustainable cloud computing: 360 degree view. ACM Computing Surveys, 51(5). https://doi.org/10.1145/3241038

Google. (2019a). Tensorflow-GPU. Https://Www.Tensorflow.Org/.

Google. (2019b). Tensorflow. Https://Www.Tensorflow.Org/.

Guan, Q., Zhang, Z., & Fu, S. (2012). A Failure Detection and Prediction Mechanism for Enhancing Dependability of Data Centers. International Journal of Computer Theory and Engineering. https://doi.org/10.7763/ijcte.2012.v4.566

Guido van Rossum. (2021). Python. Https://Docs.Python.Org/3.9/.

H2O.ai. Python Interface for H2O. (2020). GitHub.

Haiyang Ding, Zhen Zhang, Shijun Qin, Violet Guo, Zihao Chang, & Ruslan Dautov. (2017). Alibaba Cluster Trace Program. GitHub.

Hannache, O., & Batouche, M. (2015). Probabilistic model for evaluating a proactive fault tolerance approach in the cloud. 10th IEEE Int. Conf. on Service Operations and 
Logistics, and Informatics, SOLI 2015 - In Conjunction with ICT4ALL 2015. https://doi.org/10.1109/SOLI.2015.7367599

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F., Wiebe, M., Peterson, P., … Oliphant, T. E. (2020). Array programming with NumPy. In Nature (Vol. 585, Issue 7825). https://doi.org/10.1038/s41586-020-2649-2

He, H., Bai, Y., Garcia, E. A., & Li, S. (2008). ADASYN: Adaptive synthetic sampling approach for imbalanced learning. Proceedings of the International Joint Conference on Neural Networks. https://doi.org/10.1109/IJCNN.2008.4633969

He, H., & Garcia, E. A. (2009). Learning from imbalanced data. IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239

Holstein, K., Vaughan, J. W., Daumé, H., Dudík, M., & Wallach, H. (2019). Improving fairness in machine learning systems: What do industry practitioners need? Conference on 

Human Factors in Computing Systems - Proceedings. https://doi.org/10.1145/3290605.3300830

Islam, T., & Manivannan, D. (2017). Predicting Application Failure in Cloud: A Machine Learning Approach. Proceedings - 2017 IEEE 1st International Conference on Cognitive Computing, ICCC 2017. https://doi.org/10.1109/IEEE.ICCC.2017.11

Jadeja, Y., & Modi, K. (2012). Cloud computing - Concepts, architecture and challenges. 2012 International Conference on Computing, Electronics and Electrical Technologies, ICCEET 2012. https://doi.org/10.1109/ICCEET.2012.6203873

Jassas, M. S., & Mahmoud, Q. H. (2019). Failure characterization and prediction of scheduling jobs in google cluster traces. 2019 IEEE 10th GCC Conference and Exhibition, GCC 2019. https://doi.org/10.1109/GCC45510.2019.1570516010

JeminaPriyadarsini, R., & Arockiam, L. (2014). Performance Evaluation of Min-Min and Max-Min Algorithms for Job Scheduling in Federated Cloud. International Journal of Computer Applications, 99(18). https://doi.org/10.5120/17477-8393

John D. Hunter. (2003). Matplotlib. Https://Matplotlib.Org/.

Kaur, H., Nori, H., Jenkins, S., Caruana, R., Wallach, H., & Wortman Vaughan, J. (2020). Interpreting Interpretability: Understanding Data Scientists’ Use of Interpretability 
Tools for Machine Learning. Conference on Human Factors in Computing Systems - Proceedings. https://doi.org/10.1145/3313831.3376219

Khodaverdian, Z., Sadr, H., Edalatpanah, S. A., & Solimandarabi, M. N. (2021). Combination of Convolutional Neural Network and Gated Recurrent Unit for Energy Aware Resource Allocation.

Klaise, J., Van Looveren, A., Vacanti, G., & Coca, A. (2021). Alibi explain: Algorithms for explaining machine learning models. Journal of Machine Learning Research, 22.

Knauth, T., & Fetzer, C. (2015). VeCycle: Recycling VM checkpoints for faster migrations. Middleware 2015 - Proceedings of the 16th Annual Middleware Conference. https://doi.org/10.1145/2814576.2814731

Krawczyk, B. (2016). Learning from imbalanced data: open challenges and future directions. In Progress in Artificial Intelligence (Vol. 5, Issue 4). https://doi.org/10.1007/s13748-016-0094-0

Kwon, Y. C., Balazinska, M., & Greenberg, A. (2008). Fault-tolerant stream processing using a distributed, replicated file system. Proceedings of the VLDB Endowment, 1(1). https://doi.org/10.14778/1453856.1453920

Levy, S., Yao, R., Wu, Y., Dang, Y., Huang, P., Mu, Z., Zhao, P., Ramani, T., Govindaraju, N., Li, X., Lin, Q., Shafriri, G. L., & Chintalapati, M. (2020). Predictive and adaptive failure mitigation to avert production cloud VM interruptions. Proceedings of the 14th USENIX Symposium on Operating Systems Design and Implementation, OSDI 2020.

Li, F., & Hu, B. (2019). DeepJS: Job scheduling based on deep reinforcement learning in cloud data center. ACM International Conference Proceeding Series. https://doi.org/10.1145/3335484.3335513

Li, Z., Zhang, H., Brien, L. O., Cai, R., & Flint, S. (2013). The Journal of Systems and Software On evaluating commercial Cloud services : A systematic review. The Journal of Systems & Software, 86(9).

Lin, Q., Hsieh, K., Dang, Y., Zhang, H., Sui, K., Xu, Y., Lou, J. G., Li, C., Wu, Y., Yao, R., Chintalapati, M., & Zhang, D. (2018). Predicting node failure in cloud service systems. ESEC/FSE 2018 - Proceedings of the 2018 26th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering. https://doi.org/10.1145/3236024.3236060

Lin, Y., Barker, A., & Ceesay, S. (2020). Exploring Characteristics of Inter-cluster Machines and Cloud Applications on Google Clusters. Proceedings - 2020 IEEE International Conference on Big Data, Big Data 2020. https://doi.org/10.1109/BigData50022.2020.9377802

Liu, N., Li, Z., Xu, J., Xu, Z., Lin, S., Qiu, Q., Tang, J., & Wang, Y. (2017). A Hierarchical Framework of Cloud Resource Allocation and Power Management Using Deep Reinforcement Learning. 2017 IEEE 37th International Conference on Distributed Computing Systems (ICDCS), 372–382. https://doi.org/10.1109/ICDCS.2017.123

Lorido-Botran, T., Miguel-Alonso, J., & Lozano, J. A. (2014). A Review of Auto-scaling Techniques for Elastic Applications in Cloud Environments. Journal of Grid Computing, 12(4). https://doi.org/10.1007/s10723-014-9314-7

Lu, C., Ye, K., Xu, G., Xu, C. Z., & Bai, T. (2017). Imbalance in the cloud: An analysis on Alibaba cluster trace. Proceedings - 2017 IEEE International Conference on Big Data, Big Data 2017, 2018-Janua. https://doi.org/10.1109/BigData.2017.8258257

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 2017-December.

Luo, C., Zhao, P., Qiao, B., Wu, Y., Zhang, H., Wu, W., Lu, W., Dang, Y., Rajmohan, S., Lin, Q., & Zhang, D. (2021). NTAM: Neighborhood-temporal attention model for disk failure prediction in cloud platforms. The Web Conference 2021 - Proceedings of the World Wide Web Conference, WWW 2021. https://doi.org/10.1145/3442381.3449867

Manju, B. R., & Nair, A. R. (2019). Classification of Cardiac Arrhythmia of 12 Lead ECG Using Combination of SMOTEENN, XGBoost and Machine Learning Algorithms. Proceedings of the 2019 International Symposium on Embedded Computing and System Design, ISED 2019. https://doi.org/10.1109/ISED48680.2019.9096244

Matthew Rocklin. (2015). Dask. Https://Docs.Dask.Org/En/Stable/.

Mell, P., & Grance, T. (2011). The NIST definition of cloud computing. In Cloud Computing and Government: Background, Benefits, Risks. https://doi.org/10.1016/b978-0-12-804018-8.15003-x

Melo, M., Araujo, J., Matos, R., Menezes, J., & Maciel, P. (2013). Comparative analysis of migration-based rejuvenation schedules on cloud availability. Proceedings - 2013 IEEE International Conference on Systems, Man, and Cybernetics, SMC 2013. https://doi.org/10.1109/SMC.2013.701

Miller, T. (2019). Explanation in artificial intelligence: Insights from the social sciences. In Artificial Intelligence (Vol. 267). https://doi.org/10.1016/j.artint.2018.07.007

Mishra, S. K., Sahoo, B., & Parida, P. P. (2020). Load balancing in cloud computing: A big picture. In Journal of King Saud University - Computer and Information Sciences (Vol. 32, Issue 2). https://doi.org/10.1016/j.jksuci.2018.01.003

Mukwevho, M. A., & Celik, T. (2021). Toward a Smart Cloud: A Review of Fault-Tolerance Methods in Cloud Systems. IEEE Transactions on Services Computing, 14(2). https://doi.org/10.1109/TSC.2018.2816644

Outright Systems. (2019). Cloud Computing in Business. Medium.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, 

D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12.

Pezoa, J. E., & Hayat, M. M. (2014). Reliability of heterogeneous distributed computing systems in the presence of correlated failures. IEEE Transactions on Parallel and Distributed Systems, 25(4). https://doi.org/10.1109/TPDS.2013.78

Piotr Piatyszek, & Przemyslaw Biecek. (2020). Arena: universal dashboard for model exploration. Drwhy.Ai.

Plesky, E. (2019, October 24). IaaS vs PaaS vs SaaS – cloud service models compared. Plesk.

Puthal, D., Sahoo, B. P. S., Mishra, S., & Swain, S. (2015). Cloud computing features, issues, and challenges: A big picture. Proceedings - 1st International Conference on Computational Intelligence and Networks, CINE 2015. https://doi.org/10.1109/CINE.2015.31

Python. (2008). Joblib. Https://Joblib.Readthedocs.Io/En/Latest/.

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why should i trust you?” Explaining the predictions of any classifier. Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 13-17-August-2016. https://doi.org/10.1145/2939672.2939778

Rosa, A., Chen, L. Y., & Binder, W. (2015). Understanding the Dark Side of Big Data Clusters: An Analysis beyond Failures. Proceedings of the International Conference on Dependable Systems and Networks, 2015-Septe. https://doi.org/10.1109/DSN.2015.37

Rustogi, R., & Prasad, A. (2019). Swift imbalance data classification using SMOTE and extreme learning machine. ICCIDS 2019 - 2nd International Conference on Computational Intelligence in Data Science, Proceedings. https://doi.org/10.1109/ICCIDS.2019.8862112

Sak, H., Senior, A., & Beaufays, F. (2014). Long short-term memory recurrent neural network architectures for large scale acoustic modeling. Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH. https://doi.org/10.21437/interspeech.2014-80

Saleiro, P., Rodolfa, K. T., & Ghani, R. (2020). Dealing with Bias and Fairness in Data Science Systems: A Practical Hands-on Tutorial. Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. https://doi.org/10.1145/3394486.3406708

Sharma, Y., Javadi, B., Si, W., & Sun, D. (2016). Reliability and energy efficiency in cloud computing systems: Survey and taxonomy. In Journal of Network and Computer Applications (Vol. 74). https://doi.org/10.1016/j.jnca.2016.08.010

Shetty, J., Sajjan, R., & Shobha, G. (2019). Task resource usage analysis and failure prediction in cloud. Proceedings of the 9th International Conference On Cloud Computing, Data Science and Engineering, Confluence 2019. https://doi.org/10.1109/CONFLUENCE.2019.8776612

Singh, A., & Purohit, A. (2015). A Survey on Methods for Solving Data Imbalance Problem for Classification. International Journal of Computer Applications, 127(15). https://doi.org/10.5120/ijca2015906677

Soualhia, M., Khomh, F., & Tahar, S. (2015). Predicting scheduling failures in the cloud: A case study with google clusters and hadoop on Amazon EMR. Proceedings - 2015 IEEE 17th International Conference on High Performance Computing and Communications, 2015 IEEE 7th International Symposium on Cyberspace Safety and Security and 2015 IEEE 12th 
International Conference on Embedded Software and Systems, H, 58–65. https://doi.org/10.1109/HPCC-CSS-ICESS.2015.170

Sung, C., Zhang, B., Higgins, C. Y., & Choe, Y. (2016). Data-driven sales leads prediction for everything-as-a-service in the cloud. Proceedings - 3rd IEEE International Conference on Data Science and Advanced Analytics, DSAA 2016. https://doi.org/10.1109/DSAA.2016.83

Surbiryala, J., & Rong, C. (2019). Cloud computing: History and overview. Proceedings - 2019 3rd IEEE International Conference on Cloud and Fog Computing Technologies and Applications, Cloud Summit 2019. https://doi.org/10.1109/CloudSummit47114.2019.00007

Tang, H., Li, Y., Jia, T., & Wu, Z. (2016). Hunting Killer Tasks for Cloud System through Machine Learning: A Google Cluster Case Study. Proceedings - 2016 IEEE International Conference on Software Quality, Reliability and Security, QRS 2016. https://doi.org/10.1109/QRS.2016.11

Tirmazi, M., Barker, A., Deng, N., Haque, M. E., Qin, Z. G., Hand, S., Harchol-Balter, M., & Wilkes, J. (2020). Borg: The next generation. Proceedings of the 15th European Conference on Computer Systems, EuroSys 2020. https://doi.org/10.1145/3342195.3387517

Travis Oliphant. (2005). Numpy. Https://Numpy.Org/Doc/Stable/.
Verma, A., Pedrosa, L., Korupolu, M., Oppenheimer, D., Tune, E., & Wilkes, J. (2015). Large-scale cluster management at Google with Borg. Proceedings of the 10th European Conference on Computer Systems, EuroSys 2015. https://doi.org/10.1145/2741948.2741964

Verma, S., & Rubin, J. (2018). Fairness definitions explained. Proceedings - International Conference on Software Engineering. https://doi.org/10.1145/3194770.3194776

Vishwanath, K. V., & Nagappan, N. (2010). Characterizing cloud computing hardware reliability. Proceedings of the 1st ACM Symposium on Cloud Computing, SoCC ’10. https://doi.org/10.1145/1807128.1807161

Watanabe, Y., Otsuka, H., Sonoda, M., Kikuchi, S., & Matsumoto, Y. (2012). Online failure prediction in cloud datacenters by real-time message pattern learning. CloudCom 2012 - Proceedings: 2012 4th IEEE International Conference on Cloud Computing Technology and Science. https://doi.org/10.1109/CloudCom.2012.6427566

Wes McKinney. (2008). Pandas. Https://Pandas.Pydata.Org/Docs/.

Wilkes, J., Reiss, C., Deng, N., Haque, E. M., & Tirmazi, M. (2020). Google cluster-usage traces v3.

Wu, H., Zhang, W., Xu, Y., Xiang, H., Huang, T., Ding, H., & Zhang, Z. (2019). Aladdin: Optimized maximum flow management for shared production clusters. Proceedings - 2019 IEEE 33rd International Parallel and Distributed Processing Symposium, IPDPS 2019. https://doi.org/10.1109/IPDPS.2019.00078

Xu, Y., Sui, K., Yao, R., Zhang, H., Lin, Q., Dang, Y., Li, P., Jiang, K., Zhang, W., Lou, J. G., Chintalapati, M., & Zhang, D. (2020). Improving service availability of cloud systems by predicting disk error. Proceedings of the 2018 USENIX Annual Technical Conference, USENIX ATC 2018.

Yan, B., Zhao, Y., Li, Y., Yu, X., Zhang, J., & Yilin, H. Z. (2018). First Demonstration of Imbalanced Data Learning-Based Failure Prediction in Self-Optimizing Optical Networks with Large Scale Field Topology. Asia Communications and Photonics Conference, ACP, 2018-Octob. https://doi.org/10.1109/ACP.2018.8595733

Yassir, S., Mostapha, Z., & Tadonki, C. (2018). Analyzing fault tolerance mechanism of Hadoop Mapreduce under different type of failures. 2018 4th International Conference on Cloud Computing Technologies and Applications, Cloudtech 2018. https://doi.org/10.1109/CloudTech.2018.8713332

Yigitbasi, N., Gallet, M., Kondo, D., Iosup, A., & Epema, D. (2010). Analysis and modeling of time-correlated failures in large-scale distributed systems. Proceedings - IEEE/ACM International Workshop on Grid Computing. https://doi.org/10.1109/GRID.2010.5697961

Yuan, Y., Wu, Y., Wang, Q., Yang, G., & Zheng, W. (2012). Job failures in High Performance Computing Systems: A large-scale empirical study. Computers and Mathematics with Applications, 63(2). https://doi.org/10.1016/j.camwa.2011.07.040

Yuan, Z., & Zhao, P. (2019). An improved ensemble learning for imbalanced data classification. Proceedings of 2019 IEEE 8th Joint International Information Technology and Artificial Intelligence Conference, ITAIC 2019. https://doi.org/10.1109/ITAIC.2019.8785887

Zeng, M., Zou, B., Wei, F., Liu, X., & Wang, L. (2016). Effective prediction of three common diseases by combining SMOTE with Tomek links technique for imbalanced medical data. Proceedings of 2016 IEEE International Conference of Online Analysis and Computing Science, ICOACS 2016. https://doi.org/10.1109/ICOACS.2016.7563084

Zhang, J., & Mani, I. (2003). KNN Approach to Unbalanced Data Distributions: A Case Study Involving Information Extraction. Proceedings of the ICML’2003 Workshop on Learning from Imbalanced Datasets.
