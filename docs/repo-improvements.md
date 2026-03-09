# Repository improvement notes

This document captures high-impact improvements for this blog repository.

## Completed in this update

1. Added a real site description in `_config.yml` so SEO metadata, feed descriptions, and footer text are populated.
2. Added a `minima.social_links` configuration so social icons render when the footer is enabled.
3. Added a GitHub Actions workflow to build the site on push/PR and catch breakages early.

## Recommended next steps

1. **Decide whether to enable the footer**
   - `_layouts/base.html` currently comments out `footer.html`.
   - Re-enabling it would expose subscribe/social links and the site description.

2. **Pin third-party CSS dependency version**
   - `_includes/head.html` references Font Awesome with `@latest`.
   - Pinning to a specific version improves reproducibility and avoids surprise visual regressions.

3. **Add basic content quality checks**
   - Add optional markdown linting and link-checking to CI to catch broken links and style issues.

4. **Add a short contribution guide**
   - A `CONTRIBUTING.md` with post front-matter conventions and local preview steps would make updates easier.
