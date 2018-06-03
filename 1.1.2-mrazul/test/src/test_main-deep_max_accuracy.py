if __name__ == "__main__":

    from gdayf.core.controller import Controller

    #Analysis
    controller = Controller()
    if controller.config_checks():

        #Prediction
        source_data = list()
        source_data.append("/Data/Data/datasheets/binary/FODSET/")
        source_data.append("football.train2-r.csv")
        model_source = list()

        #controller = Controller()
        path_old ='/PoC_gDayF/PoC_gDayF_football.train2-r.csv_1507708316.9122167/H2ODeepLearningEstimator_1507709036.0952766'

        _, load_ar_metadata = controller.get_ar_from_engine(path_old)
        prediction_frame = controller.exec_prediction(datapath=''.join(source_data),
                                                      model_file=path_old)
        print(prediction_frame[['HomeWin', 'predict', 'p0', 'p1']])


        controller.clean_handlers()
    del controller