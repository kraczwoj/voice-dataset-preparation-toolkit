# GitHub publishing checklist

## Repository settings

Suggested repository name:

`voice-dataset-preparation-toolkit`

Suggested description:

`Local-first Python toolkit for voice dataset QC, trimming, segmentation, loudness normalization and reproducible manifests.`

Suggested topics:

`audio` `python` `voice-ai` `voice-cloning` `audio-processing` `dsp` `dataset` `quality-control` `speech` `sound-engineering`

## Before the first push

1. Replace `YOUR-USERNAME` in the README clone URL with the actual GitHub username.
2. Confirm the author name, website, and contact address.
3. Run `python -m unittest discover -s tests -v`.
4. Generate the demo dataset and run both CLI commands from the README.
5. Confirm that no WAV files, client assets, credentials, `.env` files, or local manifests are staged.

## Suggested first publication

```bash
git init
git add .
git status
git commit -m "Release Voice Dataset Preparation Toolkit 1.0.0"
git branch -M main
git remote add origin https://github.com/kraczwoj/voice-dataset-preparation-toolkit.git
git push -u origin main
```

Do not backdate commits or manufacture development history. A transparent, complete 1.0.0 release is stronger than artificial activity.

## Profile presentation

Pin the repository and use a concise profile bio such as:

`Audio Engineer, Sound Designer and Audio Experience Designer building Python tools for voice production, audio quality control and DAW automation.`

Link the profile to `https://kraczewski.studio`.
