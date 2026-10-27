"""Check installed requirements, preserving one documented upstream NCCL override."""

import importlib.metadata
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def permitted_nccl_override(package, package_version, requirement, installed_version):
    # The pinned official vLLM image intentionally overrides torch's NCCL pin:
    # DeepEPv2 GIN is compiled against NCCL >=2.30.4. Do not downgrade that .so.
    return (
        canonicalize_name(package) == "torch"
        and package_version.split("+")[0] == "2.13.0"
        and canonicalize_name(requirement.name) == "nvidia-nccl-cu13"
        and str(requirement.specifier) == "==2.29.7"
        and installed_version == "2.30.7"
    )


def main():
    # Ubuntu also exposes older apt metadata behind /usr/local packages.
    # Match import precedence instead of checking shadowed distributions.
    primary = {}
    for distribution in importlib.metadata.distributions():
        primary.setdefault(canonicalize_name(distribution.metadata["Name"]), distribution)
    distributions = list(primary.values())
    installed = {canonicalize_name(d.metadata["Name"]): d.version for d in distributions}
    errors = []
    overrides = []
    for distribution in distributions:
        name = distribution.metadata["Name"]
        for raw in distribution.requires or []:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            version = installed.get(canonicalize_name(requirement.name))
            if version is None or not requirement.specifier.contains(version, prereleases=True):
                if permitted_nccl_override(name, distribution.version, requirement, version):
                    overrides.append(f"{name}: {requirement.name}=={version} (upstream DeepEPv2 override)")
                else:
                    errors.append(f"{name} requires {raw}; installed {version or 'MISSING'}")
    if errors:
        raise SystemExit("Dependency conflicts:\n" + "\n".join(errors))
    print(f"Requirements checked for {len(distributions)} packages.")
    for override in overrides:
        print("Documented base-image override:", override)


if __name__ == "__main__":
    main()
