# DayF Core Development Framework

> **⚠️ PROJECT STATUS: CLOSED**  
> This project is no longer actively maintained or developed. It has been archived and is provided for reference purposes only.

DayF (Decision at your Fingertips) is an AutoML GPL3 opensource development framework that let developers works with Machine Learning models without any idea of AI, simply taking a .csv dataset and the objective column.

gDayF Framework make all transformations (Normalization, cleaning, etc ) and choose the  best model and parametrization selection for you storing all dataset and model execution parameters in a .json file.

## Project Status
**This project is closed and archived.** No new features, bug fixes, or updates will be provided. The codebase is preserved for historical reference and educational purposes.

## Getting Started
> **Note:** This project is archived and no longer actively maintained.

Clone Git repository: https://github.com/e2its/gdayf-core.git

## Prerequisites

### Create a virtual env (gdaf-core):
* python (3.7)
* activate gdayf-core

### Install package dependencies:
* pip install h2o==3.30.0.1
* pip install pyspark==2.4.5
* pip install pandas
* pip install hdfs
* pip install pymongo

## Docker images for ML frameworks and mongodb
* e2its/ubuntu-spark:2.4.5
* e2its/ububtu-h2o:3.30.0.1
* e2its/mongodb:latest

### Define storage parameters [Configuration can be changed on config.json]:
* MongoDB: installed on 0.0.0.0:27017:
  * "mongoDB": { "value": "gdayf-v1",
        "url": "0.0.0.0",
        "port": "27017",
        "type":"mongoDB",
        "hash_value": null, "hash_type":"MD5"
      }
* HDFS:
  * "hdfs": {"value": "/gdayf-v1/experiments" , "type":"hdfs",
        "url":"http://0.0.0.0:50070",
        "uri":"hdfs:/<<namenode_ip>>:8020",
        "hash_value": null, "hash_type":"MD5"
      }
* LocalFS:
   * "localfs": {"value": "/Data/gdayf-v1/experiments" , "type":"localfs",
        "hash_value": null, "hash_type":"MD5"
      }

* Define primary path to be used:
    * "primary_path": "localfs"


* Establish different levels of storage based on Storage engines configured:
    * "load_path": [
      {"value": "models" , "type":"localfs",
        "hash_value": null, "hash_type":"MD5"
      }
    ]
    * "log_path" : [
      {"value": "log" , "type":"localfs",
        "hash_value": null, "hash_type":"MD5"
      }
    ]
    * "json_path" : [
      {"value": "json" , "type":"mongoDB",
        "hash_value": null, "hash_type":"MD5"
      }
    ]
    * "prediction_path" : [
      {"value": "prediction" , "type":"mongoDB",
        "hash_value": null, "hash_type":"MD5"
      }
     ]

## Documentation
  A doxygen graphviz technical documentation can be located on doc folder in the project

> **Note:** Documentation may be outdated as this project is no longer maintained.

## Running the tests
`Test.py` scripts can be found on test/src folder in the project

> **Note:** Tests may not work with current versions of dependencies as this project is archived.

## Built With
  * [H2o.ai](http://http://docs.h2o.ai/) - a Machine Learning engine working on Hadoop/Yarn, Spark, or your laptop.
  * [Apache Spark MLlib](https://spark.apache.org/docs/2.4.5/) -  is a fast and general-purpose cluster computing for machine learning.
  * [mongoDB](https://docs.mongodb.com/) - NoSQL, Json based database.
  * [Apache HDFS](http://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html#Introduction) - is a distributed file system designed to run on commodity hardware.
  * [Pandas](https://pandas.pydata.org/) - is an open source Python Data Analysis Library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.

## Authors  
* **Jose L. Sanchez del Coso** - *e2its* - [Linkedin](http://www.linkedin.com/in/jlsdc)

## LICENSE
This project is licensed under the GPL3 License - see the [LICENSE.md](https://www.gnu.org/licenses/gpl-3.0.txt) for details

---

**⚠️ IMPORTANT:** This project is archived and closed. For active AutoML frameworks, please consider alternatives such as:
- [Auto-Sklearn](https://automl.github.io/auto-sklearn/)
- [H2O AutoML](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)
- [TPOT](https://epistasislab.github.io/tpot/)
- [AutoGluon](https://auto.gluon.ai/)
