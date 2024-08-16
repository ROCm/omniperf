# Maintainers Guide to Omniperf

## Publishing a release

Before publishing a new Omniperf release, please review this checklist to ensure all prerequisites are met:

1) **Ensure [VERSION](VERSION) file is updated** to reflect your desired release version.
2) **Sync `amd-mainline` with `amd-staging`**. A rebase may be required to pull all of the desired patches from the development branch to our stable mainline. Click [here](https://github.com/ROCm/omniperf/compare/amd-mainline...amd-staging) to begin that process.
3) **Update [CHANGES](CHANGES)** to reflect all major modifications to the codebase since the last release. When modifying [CHANGES](CHANGES) please ensure formatting is consistent with the rest of the ROCm software stack. See [this template](https://github.com/ROCm/hipTensor/blob/develop/CHANGELOG.md) for reference.
4) **Confirm all CI tests are passing**. You can easily confirm this by peeking the passing status of all GitHub continuous integration tests.
5) **Create a tag from `amd-mainline`**. More information on tagging can be found at [Git Docs - Tagging](https://git-scm.com/book/en/v2/Git-Basics-Tagging). 

> [!NOTE]
Note: A successful tag should trigger the [packaging action](https://github.com/ROCm/omniperf/actions/workflows/packaging.yml) which will produce a tarball artifact. **This artifact needs to be included as an asset in your release**.

Once you've completed the above checklist, you are ready to publish your release. Please ensure you follow formatting from [past Omniperf releases](https://github.com/ROCm/omniperf/releases) for consistency. Some important aspects of our release formatting include:

- Date of release is included in "Release Title".
- Updates are called out in "Release Description". Updates should mirror those listed in [CHANGES](CHANGES).
- Links to documentation and associated release tarball are called out in "Release Description".
- The tarball artifact from the corresponding tag is added to "Release Assets".
