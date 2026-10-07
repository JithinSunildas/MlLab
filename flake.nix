{
  description = "Python ML flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in
    {
      devShells.${system}.default =
        (pkgs.buildFHSEnv {
          name = "ml-python-env";

          targetPkgs =
            pkgs:
            (with pkgs; [
              (python3.withPackages (
                ps: with ps; [
                  # Notebooks & Kernels
                  jupyter
                  notebook
                  ipykernel
                  ipywidgets
                  tensorflow-lite

                  # Math & Data
                  numpy
                  scipy
                  pandas

                  # Machine Learning
                  scikit-learn

                  # Visualization
                  matplotlib
                  seaborn

                  # Deep Learning
                  torch
                  torchvision

                  # Utilities
                  tqdm
                  pillow
                  requests
                  ruff
                  black
                  isort
                  openpyxl

                  # Virtualenv / Pip support inside FHS
                  pip
                  virtualenv
                ]
              ))

              # Essential native runtime libraries
              libGL
              glib
              zlib
              stdenv.cc.cc.lib
              fontconfig
              freetype

              # Flattened X11 libraries (replaces deprecated xorg.* names)
              libx11
              libxext
              libxrender
              libice
              libsm
            ]);

          runScript = "fish";
        }).env;
    };
}
