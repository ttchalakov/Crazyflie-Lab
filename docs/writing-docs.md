# Writing docs

The site is built by [Zensical](https://zensical.org) from the Markdown in `docs/`,
configured in `zensical.toml`.

## Preview

```bash
zensical serve
```

This serves the site locally and reloads as you edit.

## Add a page

Create a Markdown file under `docs/` and add it to `nav` in `zensical.toml`.

## Images and GIFs

Put media in `docs/assets/` and link it relative to the page:

```markdown
![Drone at hover](assets/hover.gif)
![Wiring](assets/wiring.png){ width="400" }
```

`{ width="400" }` sets the displayed size. For a caption, wrap the image in a
figure:

```markdown
<figure markdown>
  ![Drone at hover](assets/hover.gif)
  <figcaption>Hovering on the neural policy.</figcaption>
</figure>
```

## Math

Write LaTeX between `$...$` inline or `$$...$$` for a display block:

```markdown
The thrust acts along the body z-axis, $T\,R(\eta)\,e_3$:

$$
m\,\dot v = -m g\,e_3 + T\,R(\eta)\,e_3
$$
```

## Publishing

`.github/workflows/docs.yml` builds the site and publishes it to GitHub Pages when
a release tag (`vX.Y.Z`) is pushed. It can also be run by hand from the Actions
tab. In the repository settings, Pages must use **GitHub Actions** as its source.
