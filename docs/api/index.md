# API Reference

Technical reference for the `lst_tools` package.

## Modules

- [Grid & Flow](core.md): Core data structures for grid and flow data
- [Config](config.md): Configuration management
- [Data I/O](data-io.md): File readers and writers
- [Geometry](geometry.md): Surface geometry computations
- [Convert](convert.md): Format conversion utilities
- [Setup](setup.md): LST calculation setup (parsing, tracking, spectra)
- [Processing](processing.md): Post-processing of LST results
- [HPC](hpc.md): HPC job script generation

## Package Architecture

``` mermaid
graph LR

    %% leaf modules
    data_io["data_io<br/><small>Fortran binary, Tecplot, LASTRAC</small>"]
    geometry["geometry<br/><small>curvature, surface angle, radius</small>"]
    config["config<br/><small>read, write, validate</small>"]

    %% intermediate modules
    core["core<br/><small>Grid, Flow</small>"]
    convert["convert<br/><small>meanflow, LST input</small>"]

    %% workflow modules
    setup["setup<br/><small>parsing, tracking, spectra</small>"]
    process["process<br/><small>tracking, spectra</small>"]
    hpc["hpc<br/><small>job scripts, scheduler</small>"]
    cli["cli<br/><small>command-line interface</small>"]

    %% dependencies
    data_io --> core
    convert --> core
    convert --> data_io
    setup --> config
    setup --> convert
    process --> data_io
    process --> config
    cli --> setup
    cli --> process
    cli --> convert
    cli --> hpc
    cli --> config
```
