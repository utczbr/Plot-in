# Third-Party Software Notices and Information

This project, **LYAA-fine-tuning**, incorporates third-party software components and assets under their respective open-source licenses. This document provides licensing notices, copyright information, and inventories of third-party assets included in this distribution.

---

## 1. Font Awesome Free 7.1.0

- **Project:** Font Awesome Free
- **Version:** 7.1.0
- **Creator:** Fonticons, Inc. (@fontawesome)
- **Website:** [https://fontawesome.com](https://fontawesome.com)
- **License Information:** [https://fontawesome.com/license/free](https://fontawesome.com/license/free)
- **Copyright:** Copyright (c) 2026 Fonticons, Inc.

### Licensing Summary

Font Awesome Free is provided under a triple license depending on the asset type:

1. **Icons (SVG and JS formats):**
   - **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
   - **URL:** [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)
   - In the Font Awesome Free download, the CC BY 4.0 license applies to all icons packaged as SVG and JS file types.

2. **Fonts (Web fonts, TTF, WOFF, WOFF2):**
   - **License:** SIL Open Font License 1.1 (SIL OFL 1.1)
   - **URL:** [https://openfontlibrary.org/en/license/sil-open-font-license-ofl](https://openfontlibrary.org/en/license/sil-open-font-license-ofl)
   - In the Font Awesome Free download, the SIL OFL 1.1 license applies to all font files.

3. **Code (CSS, LESS, SCSS, JavaScript):**
   - **License:** MIT License
   - **URL:** [https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT)
   - In the Font Awesome Free download, the MIT license applies to all non-font, non-icon code files.

### Attribution Notice

All shipped SVG icons retain the embedded Font Awesome attribution header:
```xml
<!--!Font Awesome Free 7.1.0 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2026 Fonticons, Inc.-->
```

---

## 2. Shipped SVG Icons Inventory (`src/ui/rendering/icons/`)

All icons in `src/ui/rendering/icons/` are sourced from Font Awesome Free 7.1.0, normalized to a `0 0 640 640` viewBox with optical centering.

| Icon Filename | Upstream Glyph | Style | ViewBox | License | Role / Application Usage |
|---|---|---|---|---|---|
| `bullseye-solid-full.svg` | `bullseye` | Solid | `0 0 640 640` | CC BY 4.0 | Action button icon for "Run Batch Analysis" |
| `check-solid-full.svg` | `check` | Solid | `0 0 640 640` | CC BY 4.0 | Status indicator for reviewed / cached files |
| `circle-chevron-left-solid-full.svg` | `circle-chevron-left` | Solid | `0 0 640 640` | CC BY 4.0 | Navigation button ("Previous Image") & Sidebar toggle |
| `circle-chevron-right-solid-full.svg` | `circle-chevron-right` | Solid | `0 0 640 640` | CC BY 4.0 | Navigation button ("Next Image") & Sidebar toggle |
| `floppy-disk-solid-full.svg` | `floppy-disk` | Solid | `0 0 640 640` | CC BY 4.0 | Action button icon for "Save & Next" protocol results |
| `folder-open-solid-full.svg` | `folder-open` | Solid | `0 0 640 640` | CC BY 4.0 | Directory input selector ("Browse Input / Models") |
| `gear-solid-full.svg` | `gear` | Solid | `0 0 640 640` | CC BY 4.0 | Toolbar launcher icon for Advanced Settings dialog |
| `image-solid-full.svg` | `image` | Solid | `0 0 640 640` | CC BY 4.0 | Default file list item icon (pending / unreviewed image) |
| `repeat-solid-full.svg` | `repeat` | Solid | `0 0 640 640` | CC BY 4.0 | Action button icon for "Recalibrate" axes flow |

### Planned Wave 1 Icon Deliverables (T-018.8 / ADR-018)

In accordance with ADR-018, the following additional Font Awesome Free 7.1.0 glyphs (centered inside `<g transform="translate(64, 64)">` normalized to `640x640`) are scheduled for delivery in Wave 1:
- `triangle-exclamation-solid-full.svg` — Warning indicator for low-confidence images / needs-review items (replaces missing `warning-solid.svg`).
- State set: `circle-regular`, `circle-solid`, `xmark-solid`, `circle-question-regular`.
- Editor & sidebar set: `eye`, `pen-to-square`, `draw-polygon`, `location-crosshairs`, `chart-pie`, `ruler-combined`, `arrow-rotate-left`, `arrow-rotate-right`, `trash-can`, `ellipsis`, `keyboard`.

---

## 3. License Texts

### 3.1 Creative Commons Attribution 4.0 International (CC BY 4.0)

Creative Commons Corporation ("Creative Commons") is not a law firm and does not provide legal services or legal advice. Distribution of Creative Commons public licenses does not create a lawyer-client or other relationship.

#### Summary of Rights:
You are free to:
- **Share** — copy and redistribute the material in any medium or format for any purpose, even commercially.
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially.
- The licensor cannot revoke these freedoms as long as you follow the license terms.

#### Under the Following Terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **No additional restrictions** — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

For the full legal code of CC BY 4.0, please visit:
[https://creativecommons.org/licenses/by/4.0/legalcode](https://creativecommons.org/licenses/by/4.0/legalcode)

---

### 3.2 SIL Open Font License 1.1 (SIL OFL 1.1)

Copyright (c) 2026 Fonticons, Inc. (https://fontawesome.com)
with Reserved Font Name "Font Awesome".

This Font Software is licensed under the SIL Open Font License, Version 1.1.
This license is copied below, and is also available with a FAQ at:
[https://openfontlibrary.org/en/license/sil-open-font-license-ofl](https://openfontlibrary.org/en/license/sil-open-font-license-ofl)

```
PREAMBLE
The goals of the Open Font License (OFL) are to stimulate worldwide
development of collaborative font projects, to support the font creation
efforts of academic and linguistic communities, and to provide a free and
open framework in which fonts may be shared and improved in partnership
with others.

The OFL allows the licensed fonts to be used, studied, modified and
redistributed freely as long as they are not sold by themselves. The
fonts, including any derivative works, can be bundled, embedded,
redistributed and/or sold with any software provided that any reserved
names are not used by derivative works. The fonts and derivatives,
however, cannot be released under any other type of license. The
requirement for fonts to remain under this license does not apply
to any document created using the fonts or their derivatives.

DEFINITIONS
"Font Software" refers to the set of files released by the Copyright
Holder(s) under this license and clearly marked as such. This may include
source files, build scripts and documentation.

"Reserved Font Name" refers to any names specified as such after the
copyright statement(s).

"Original Version" refers to the collection of Font Software components as
distributed by the Copyright Holder(s).

"Modified Version" refers to any derivative made by adding to, deleting,
or substituting -- in part or in whole -- any of the components of the
Original Version, by changing formats or by porting the Font Software to a
new environment.

"Author" refers to any designer, engineer, programmer, technical
writer or other person who contributed to the Font Software.

PERMISSION & CONDITIONS
Permission is hereby granted, free of charge, to any person obtaining
a copy of the Font Software, to use, study, copy, merge, embed, modify,
redistribute, and sell modified and unmodified copies of the Font
Software, subject to the following conditions:

1) Neither the Font Software nor any of its individual components,
in Original or Modified Versions, may be sold by itself.

2) Original or Modified Versions of the Font Software may be bundled,
redistributed and/or sold with any software, provided that each copy
contains the above copyright notice and this license. These can be
included either as stand-alone text files, human-readable headers or
in the appropriate machine-readable metadata fields within text or
binary files as long as those fields can be easily viewed by the user.

3) No Modified Version of the Font Software may use the Reserved Font
Name(s) unless prominent written permission is granted by the corresponding
Copyright Holder. This restriction only applies to the primary font name as
presented to the users.

4) The name(s) of the Copyright Holder(s) or the Author(s) of the Font
Software shall not be used to promote, endorse or advertise any
Modified Version, except to acknowledge the contribution(s) of the
Copyright Holder(s) and the Author(s) or with their explicit written
permission.

5) The Font Software, modified or unmodified, in part or in whole,
must be distributed entirely under this license, and must not be
distributed under any other license. The requirement for fonts to
remain under this license does not apply to any document created
using the Font Software.

TERMINATION
This license becomes null and void if any of the above conditions are
not met.

DISCLAIMER
THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
OF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL THE
COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
INCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL
DAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM
OTHER DEALINGS IN THE FONT SOFTWARE.
```

---

### 3.3 The MIT License (MIT)

Copyright (c) 2026 Fonticons, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
