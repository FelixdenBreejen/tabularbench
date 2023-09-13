import sys
from typing import Optional
from tabularbench.data.dataset_openml import OpenMLDataset
from tabularbench.sweeps.run_config import RunConfig
from tabularbench.sweeps.sweep_start import set_seed



def run_experiment(cfg: RunConfig) -> Optional[dict]:

    cfg.logger.info(f"Start experiment on {cfg.openml_dataset_name} (id={cfg.openml_dataset_id}) with {cfg.model} doing {cfg.task} with {cfg.feature_type} features")
    
    set_seed(cfg.seed)
    cfg.logger.info(f"Seed is set to {cfg.seed}, device is {str(cfg.device)}")

    cfg.logger.info(f"We are using the following hyperparameters:")
    for key, value in cfg.model_hyperparameters.items():
        cfg.logger.info(f"    {key}: {value}")



    if debugger_is_active():
        return run_experiment_(cfg)
    
    try:
        return run_experiment_(cfg)
    except Exception as e:
        # Print to the console
        cfg.logger.exception("Exception occurred while running experiment")

        # TODO: implement remove checkpoint files

        # if config["model_type"] == "skorch" and config["model__use_checkpoints"]:
        #     print("crashed, trying to remove checkpoint files")
        #     model_id = inspect.trace()[-1][0].f_locals['model_id']
        #     try:
        #         os.remove(r"skorch_cp/params_{}.pt".format(model_id))
        #     except:
        #         print("could not remove params file")
        # if config["model_type"] == "tab_survey":
        #     print("Removing checkpoint files")
        #     print("Removing ")
            
        #     model_id = inspect.trace()[-1][0].f_locals['model_id']
        #     print(r"output/saint/{}/tmp/m_{}_best.pt".format(config["data__keyword"], model_id))
        #     #try:
        #     os.remove(r"output/saint/{}/tmp/m_{}_best.pt".format(config["data__keyword"], model_id))
        #     #except:
        #     #print("could not remove params file")
        
        return None
    

    pass

def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


def run_experiment_(cfg: RunConfig) -> dict:

    
    dataset = OpenMLDataset(cfg.openml_dataset_id, cfg.task, cfg.feature_type, cfg.dataset_size)

    for x_train, x_val, x_test, y_train, y_val, y_test, categorical_indicator in dataset.split_generator():

        pass