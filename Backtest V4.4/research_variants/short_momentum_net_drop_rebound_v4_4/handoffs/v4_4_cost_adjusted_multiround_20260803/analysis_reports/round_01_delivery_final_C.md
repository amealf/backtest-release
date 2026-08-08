# Round-1 DELIVERY_FINAL — member C

## Delivery outcome

- Active campaign: `v4_4_cost_adjusted_multiround_20260803`
- Round: `round_01_broad_all_window`
- Coordinates: `372`
- Trades: `316398`
- Parameter acceptance: `none`
- Completed stage-analysis SHA-256: `3c55f1222db2586f8e5fb4dee5800e4534ee695c9ce7e101fd9a7e5f7c56d03f`

The one stage+cumulative delivery invocation completed the stage fixed-template analysis, then failed before stable publication because the full campaigns root mixed an older implementation-validation stage with a different `engine_sha256`. Its failed status and partial snapshot were preserved. The leader then authorized one cumulative-only recovery scoped exactly to the active campaign, using the existing union builder and four review workers. That recovery completed and atomically published the stable routes.

## Published cumulative closure

- Snapshot id: `2020ad7b12d57889f1c1d0cf69f981bcf2b5e3ec5b8a4808c196dbb6cdd51d47`
- Current-pointer SHA-256: `24ae69b6d57becb6931d79ee630e7b9d0a91e81821a32d38fe8ee76834a784a6`
- Analysis-manifest SHA-256: `dbaccce3bc61ea3979c2340ddd9616e82c01a0261bea25a6998ffdd2ac4534db`
- Completion-manifest SHA-256: `46f70611a95f57a7f44c22310d1d4432cc63a9e47b089090f879f760643d4a0a`
- Trade-review-manifest SHA-256: `a51dff3aa3855073712d90b9cec39c1aaf4c52be4bacb6c71b2aff17ca8293c2`
- Snapshot main HTML SHA-256: `40bf5e4b4f5531ac0bcafcfb0adf9cda8b8f2a22a65d301c079b64697a344ad3`
- Snapshot trade HTML SHA-256: `125214644c9a0eab53014f86080188ae6f7204f0d318bb535d59d44c10538ed1`
- Snapshot scenario HTML SHA-256: `c7ae6d952567e9f3783feb91e478b09626510121b73797a6c73fb966782cffd1`
- Stable main redirect SHA-256: `c8910c2f369aede55b75ead52d9b80988683c448e83ad9973a5d766d59d0ea65`
- Stable trade redirect SHA-256: `1fb63afd7816229edc093c72fadd9e249fcfc6c1dac05bf17221885399740ec5`
- Hash/size checks: `384`, mismatches: `0`

## Browser and visual QA

| Surface | QA SHA-256 | States | Runtime errors | External requests | Layout failures | Screenshots |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Stage analysis | `94b8b6fa185c26a94dfca48f480d699aee308f249aa8bc2702a6c51406da5762` | 200 | 0 | 0 | 0 | 6 |
| Published snapshot | `bf93a49733a76a87f5a237e1b04d81ef57af44caa33f1f68537a5af0f01c126d` | 200 | 0 | 0 | 0 | 6 |
| Stable redirects | `e7b9ec9e28a63ba7eb12234c8446301facf0ce0c6b865945a827a797b1cf6708` | 200 | 0 | 0 | 0 | 6 |

- Cost-adjusted default header: `成本后排名 ▲`
- Gross header after switching: `毛收益排名 ▲`
- Switching mode changes the ordering and the displayed return columns.
- Stable-route screenshots are byte-identical to snapshot screenshots for all six desktop/mobile captures.
- Manual visual review found no garbled text, unintended overlap, broken alignment, or asymmetric layout. The 372-row ranking page is intentionally long; mobile tables are contained/scrollable, and scenario/trade chart panels stack cleanly.

## Preserved evidence

- Source manifest remains `6fa3d0c8eb0277066ef5f70fca4a9fbab1d31fbb30e023cd8fd83d233192ae16`.
- Frozen plan remains `1424dc17862a2bfe0b8f0439fef061e64efc487c5057b7cff64498ed40a78046`.
- Raw completion remains `c9532f77b626f647dcfe7b1fdc09ee76b2b895e851da0b98f6206b26ff1e6539`.
- Failed delivery status remains preserved: `bc482249766613fe3d88df26e45100deb272f25214d4193c00476806bcf8f4dc`.
- Failed partial snapshot `1ba05465b49a1de45c407bfd9b4456eeb83c92b9dadea8d1c5d060a40ae22d98` remains present.
- Older temporary campaign completion remains unchanged: `eddaf0d6335b2e718e4e78d1d1e5fc06aa16a3e4bb559ef062a7c357de36770e`.
