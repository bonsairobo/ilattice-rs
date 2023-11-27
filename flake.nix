{
  description = "NixOS environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
  in {
    devShell.${system} = with pkgs;
      mkShell {
        buildInputs = with pkgs; [
          clang
          # Replace llvmPackages with llvmPackages_X, where X is the latest
          # LLVM version (at the time of writing, 16)
          llvmPackages_16.bintools
          mold
          pkg-config
          rustup
          yq-go
        ];

        shellHook = ''
          export RUSTC_VERSION=$(yq ".toolchain.channel" rust-toolchain.toml)
          export PATH=$PATH:''${CARGO_HOME:-~/.cargo}/bin
          export PATH=$PATH:''${RUSTUP_HOME:-~/.rustup}/toolchains/$RUSTC_VERSION-x86_64-unknown-linux-gnu/bin/
          rustup target add wasm32-unknown-unknown
          rustup component add rust-analyzer
        '';
      };
  };
}
