{
    description = "PBF65 Development Environment";

    inputs = {
        nixpkgs.url = "nixpkgs";
        flake-utils.url = "github:numtide/flake-utils";
        rust-overlay.url = "github:oxalica/rust-overlay";
    };

    outputs = { nixpkgs, flake-utils, rust-overlay, ... }:
        flake-utils.lib.eachDefaultSystem (system:
            let
                pkgs = import nixpkgs {
                    inherit system;
                    overlays = [ rust-overlay.overlays.default ];
                };

                runtimeLibs = with pkgs; [
                    wayland
                    libxkbcommon
                    vulkan-loader
                    libGL
                ];
            in
            {
                devShells.default = pkgs.mkShell {
                    buildInputs = with pkgs; [
                        cargo
                        cargo-watch
                        nodejs
                        wasm-pack
                        (rust-bin.stable.latest.default.override {
                            extensions = [ "rust-src" ];
                            targets = [ "wasm32-unknown-unknown" ];
                        })
                        python3
                    ] ++ runtimeLibs;

                    shellHook = ''
                        export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibs}:$LD_LIBRARY_PATH"
                    '';
                };
            }
        );
}
