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

                webgpu-browser = pkgs.ungoogled-chromium.override {
                    commandLineArgs = [
                        "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan"
                        "--enable-unsafe-webgpu"
                        "--use-vulkan"
                        "--ignore-gpu-blocklist"
                        # "--ozone-platform=x11"
                        "--user-data-dir=.chrome-webgpu-profile"
                    ];
                };

                rust = pkgs.rust-bin.stable.latest.default.override {
                    extensions = [ "rust-src" ];
                    targets = [ "wasm32-unknown-unknown" ];
                };

                serveCmd = pkgs.writeShellApplication {
                    name = "serve";
                    runtimeInputs = [ pkgs.simple-http-server ];
                    text = ''
                        simple-http-server -i --nocache --coop --coep --ip 127.0.0.1 -p 8080 "$@"
                    '';
                };

                buildCmd = pkgs.writeShellApplication {
                    name = "build-wasm";
                    runtimeInputs = [
                        pkgs.cargo
                        pkgs.cargo-watch
                        pkgs.nodejs
                        pkgs.wasm-pack
                        rust
                    ];
                    text = ''
                        wasm-pack build --target web "$@"
                    '';
                };

                devCmd = pkgs.writeShellApplication {
                    name = "dev";
                    runtimeInputs = [ buildCmd serveCmd webgpu-browser ];
                    text = ''
                        pkill -f "simple-http-server.*8080" || true
                        build-wasm
                        serve &
                        SERVER_PID=$!
                        trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
                        chromium "http://127.0.0.1:8080"
                    '';
                };
                
                mkApp = drv: {
                    type = "app";
                    program = pkgs.lib.getExe drv;
                };
            in
            {
                apps = {
                    default = mkApp devCmd;
                    dev = mkApp devCmd;
                    serve = mkApp serveCmd;
                    build = mkApp buildCmd;
                    browser = mkApp webgpu-browser;
                };

                devShells.default = pkgs.mkShell {
                    buildInputs = [
                        pkgs.cargo
                        pkgs.cargo-watch
                        pkgs.nodejs
                        pkgs.wasm-pack
                        pkgs.simple-http-server
                        rust
                        webgpu-browser
                        serveCmd
                        buildCmd
                        devCmd
                    ] ++ runtimeLibs;

                    shellHook = ''
                        export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibs}:$LD_LIBRARY_PATH"
                    '';
                };
            }
        );
}
