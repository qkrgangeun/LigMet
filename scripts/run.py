from lightning.pytorch.cli import LightningCLI  # type: ignore
from ligmet.pl import LigMetTestModule, LigMetTestDataModule  # type: ignore

def main():
    cli = LightningCLI(
        LigMetTestModule, 
        LigMetTestDataModule,
        save_config_kwargs={"overwrite": True}
    )


if __name__ == "__main__":
    main()