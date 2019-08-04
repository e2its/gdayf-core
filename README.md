# gDAYF Core
DayF (Decision at your Fingertips) is an AutoML freeware development framework that let developers works with Machine Learning models without any idea of AI, simply taking a csv dataset and the objective column.
The software make all transformation (Normalization, cleaning) and choose the  best model and parametrization selection for you.
Currently is Freeware but soon will be release as opensource.

Clone Git repository: https://github.com/e2its/gdayf-core.git

##Prerequisites:

#Create a virtual env (gdaf-core) versions below:
python: 3.5

activate gdayf-core
pip install h2o==3.20.0.8 
pip install pyspark==2.2.3
pip instal pandas==0.24.2
pip install hdfs==2.1.0
pip install pymongo==3.8.0



Define storage parameters:

1. MongoDB: installed on 127.0.0.1:33017
[Configuration can be changed on config.json]
    "mongoDB": { "value": "gdayf-v1",
        "url": "localhost",
        "port": "33017",
        "type":"mongoDB",
        "hash_value": null, "hash_type":"MD5"
      }
2.HDFS (Apache Hadoop 3.2.0:
    "hdfs": {"value": "/gdayf-v1/experiments" , "type":"hdfs",
        "url":"http://localhost:9870",
        "uri":"hdfs://localhost:9000",
        "hash_value": null, "hash_type":"MD5"
      }
3.LocalFS:
    "localfs": {"value": "/Data/gdayf-v1/experiments" , "type":"localfs",
        "hash_value": null, "hash_type":"MD5"
      }
4. Define primary path to be used:
    "primary_path": "localfs"
    
5. Establish diferent levels of storage based on Storage engines configured:
    "load_path": [
      {"value": "models" , "type":"mongoDB",
        "hash_value": null, "hash_type":"MD5"
      }
    ],
    "log_path" : [
      {"value": "log" , "type":"localfs",
        "hash_value": null, "hash_type":"MD5"
      }
    ],
    "json_path" : [
      {"value": "" , "type":"mongoDB",
        "hash_value": null, "hash_type":"MD5"
      }
    ],
    "prediction_path" : [
      {"value": "prediction" , "type":"mongoDB",
        "hash_value": null, "hash_type":"MD5"
      }
  
  
