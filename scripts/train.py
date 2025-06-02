from lightning.pytorch.cli import LightningCLI  # type: ignore
from ligmet.pl import LigMetModel, LigMetDataModule  # type: ignore

def main():
    cli = LightningCLI(
        LigMetModel, 
        LigMetDataModule,
        save_config_kwargs={"overwrite": True}
    )


if __name__ == "__main__":
    main()