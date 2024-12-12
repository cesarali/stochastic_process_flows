import argparse
from spflows.training.basic_experiments import BasicLightningExperiment
from spflows.configs_classes.forecasting_configs import ForecastingModelConfig

def train(args):
    # Update dataclass with parsed arguments
    config = ForecastingModelConfig(
        seed=args.seed,
        dataset=args.dataset,
        network=args.network,
        noise=args.noise,
        diffusion_steps=args.diffusion_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_cells=args.num_cells,
        hidden_dim=args.hidden_dim,
        residual_layers=args.residual_layers,
        num_batches_per_epoch=2,
    )
    trainer = BasicLightningExperiment(config)
    trainer.train()

if __name__=="__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description='Train forecasting model.')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default="electricity_nips")
    parser.add_argument('--network', type=str, default="timegrad_rnn", choices=[
        'timegrad', 'timegrad_old', 'timegrad_all', 'timegrad_rnn', 'timegrad_transformer', 'timegrad_cnn'
    ])
    parser.add_argument('--noise', type=str, choices=['normal', 'ou', 'gp'], default="gp")
    parser.add_argument('--diffusion_steps', type=int, default=9)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=int, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_cells', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=10)
    parser.add_argument('--residual_layers', type=int, default=2)
    args = parser.parse_args()

    train(args)
