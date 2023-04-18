import json
import os
import torch

model_params_subdir = "ModelParameters"
latent_codes_subdir = "LatentCodes"
specifications_filename = "specs.json"


def load_experiment_specifications(experiment_directory):
    filename = os.path.join(experiment_directory, specifications_filename)
    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            + '"specs.json"'.format(experiment_directory)
        )
    return json.load(open(filename))


def load_model_parameters_decoder(experiment_directory, checkpoint, decoder):
    filename = os.path.join(
        experiment_directory, model_params_subdir, checkpoint + ".pth"
    )
    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))
    data = torch.load(filename)
    decoder.load_state_dict(data["model_state_dict"])
    return data["epoch"]


def get_latent_codes_dir(experiment_dir, create_if_nonexistent=False):
    dir = os.path.join(experiment_dir, latent_codes_subdir)
    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)
    return dir


def load_latent_vectors(experiment_directory, filename):
    full_filename = os.path.join(
        get_latent_codes_dir(experiment_directory), filename
    )
    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))
    return torch.load(full_filename)["latent_codes"]

def fourier_transform(x, L=5):
    cosines = torch.cat([torch.cos(2**l*3.1415*x) for l in range(L)], -1)
    sines = torch.cat([torch.sin(2**l*3.1415*x) for l in range(L)], -1)
    transformed_x = torch.cat((cosines,sines),-1)
    return transformed_x
