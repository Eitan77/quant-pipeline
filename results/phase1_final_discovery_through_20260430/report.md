# Final Phase 1 full-pre-holdout discovery

All permitted data through 2026-04-30 was used for discovery and candidate ranking. Historical subperiod and recent-period results are robustness diagnostics. 2026-05-01 onward was not accessed.

Statistical anomaly evidence only; every promoted finding requires Phase 2 strategy and execution testing.

Regime, scope, exact-time, and Phase 2 recommendation diagnostics are descriptive only and were not used to alter FDR, candidate ranking, candidate selection, promotion, or anomaly status.

## Data and methodology

- Evidence label: full_pre_holdout_discovery
- Discovery start: 2019-06-21
- Discovery end: 2026-04-30
- Sealed holdout start: 2026-05-01
- Holdout access: false
- Source-data fingerprint: dbb97463bdfd39925cd2b6e27a3d72427c68cbeb1aa1c8160a6b9eed48618283
- Git commit: b1798f03c8cb6835910622ed3c9c37370f14ce1d
- Cache schema: phase1_final
- Separate confirmation period: false
- Primary FDR family tests: 13380
- Exploratory FDR family tests: 131148
- Primary inference: global Benjamini-Hochberg FDR across prespecified primary targets.
- Exploratory inference: separate families; exploratory and recency-weighted evidence cannot independently promote a candidate.

## Results funnel

- Features requested: 820
- Features successfully built: 798
- Targets requested: 234
- Broad-screen feature-target pairs: 186732
- Pairs passing coverage: 144528
- Primary globally significant pairs: 1185
- Exploratory significant pairs: 11421
- Redundancy clusters: 47
- Exact diagnostic candidates: 80
- Robust Phase 1 anomaly candidates: 16
- Candidates requiring Phase 2 testing: 4

## Top relationships

| feature | target | top_bottom_spread | bh_fdr_p | recent_5y_effect | recent_3y_effect | recent_2y_effect | recent_12m_effect | jan_apr_2026_effect | recent_classification | symbol_breadth_classification | phase2_recommendation | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| session_range_position | fwd_return_60m_benchmark_adjusted | 0.00020254883565939963 | 4.858555709986966e-06 | 0.0002294143196195364 | 0.00030863104620948434 | 0.0003813548246398568 | 0.0004233398358337581 | 0.0006453354144468904 | strengthening_recently | broad_across_symbols | retain_for_monitoring | robust_phase1_anomaly_candidate |
| session_range_position | fwd_return_30m_benchmark_adjusted | 0.00014132440992398188 | 3.499544714609299e-07 | 0.00015998180606402457 | 0.00018601883493829519 | 0.00023345505178440362 | 0.000269364973064512 | 0.00036251640995033085 | strengthening_recently | broad_across_symbols | retain_for_monitoring | robust_phase1_anomaly_candidate |
| close_location | fwd_return_120m_benchmark_adjusted | 0.00012463778693927452 | 1.2177061222850438e-07 | 0.0001491736329626292 | 0.00018970131350215524 | 0.00022651749895885587 | 0.00021695057512260973 | 0.0004165229620411992 | strengthening_recently | broad_across_symbols | retain_for_monitoring | robust_phase1_anomaly_candidate |
| return_consistency_5 | fwd_return_120m_benchmark_adjusted | 0.00018001820717472583 | 0.00045087458549196555 | 0.00018914532847702503 | 0.00018076677224598825 | 0.00016972202865872532 | 0.00019337478443048894 | 0.0005518895341083407 | persistent | broad_across_symbols | retain_for_monitoring | requires_phase2_testing |
| range_position_1 | fwd_return_120m_benchmark_adjusted | 0.00012463778693927452 | 1.2177061222850438e-07 | 0.0001491736329626292 | 0.00018970131350215524 | 0.00022651749895885587 | 0.00021695057512260973 | 0.0004165229620411992 | strengthening_recently | broad_across_symbols | retain_for_monitoring | robust_phase1_anomaly_candidate |
| market_residual_return_60 | fwd_return_30m_beta_residual | -0.00014925121286069043 | 5.165496756682678e-05 | 0.0001485894899815321 | 0.00010097214544657618 | 9.763706475496292e-05 | 2.5729739718372002e-05 | 2.7211759515921585e-05 | weakening_recently | broad_across_symbols | retain_for_monitoring | robust_phase1_anomaly_candidate |
| range_position_10_z_4680 | fwd_return_120m_benchmark_adjusted | 0.00019878407692885958 | 0.0007132391429447286 | 0.0002228202938567847 | 0.0002802816452458501 | 0.0003153416037093848 | 0.0002729315310716629 | 0.00044423312647268176 | persistent | broad_across_symbols | retain_for_monitoring | robust_phase1_anomaly_candidate |
| market_residual_return_20 | fwd_return_5m_beta_residual | -7.492187432944775e-05 | 8.919430097357605e-10 | 7.050670683383942e-05 | 7.338803698075935e-05 | 6.263334944378585e-05 | 2.388781649642624e-05 | 4.7791661927476525e-05 | weakening_recently | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |
| lower_high | fwd_return_120m_benchmark_adjusted | -0.00014788655971642584 | 0.0006035888364767113 | 0.00016144063556566834 | 0.00021946729975752532 | 0.00018968665972352028 | 0.0002813060418702662 | 0.00041562208207324147 | strengthening_recently | moderately_concentrated | retain_for_monitoring | requires_phase2_testing |
| market_residual_return_60 | fwd_return_eod_beta_residual | -0.00020356354798423126 | 4.390080531196736e-08 | 0.00018507425556890666 | 0.00013512276927940547 | 0.00013604336709249765 | 7.85045194788836e-05 | 0.00010569571168161929 | weakening_recently | broad_across_symbols | retain_for_monitoring | robust_phase1_anomaly_candidate |
| opening_close_location_20m | fwd_return_30m_benchmark_adjusted | 0.00010280899005010724 | 0.003226792078589122 | 0.0001191165647469461 | 0.00017438395298086107 | 0.0001755828852765262 | 0.00016701575077604502 | 0.000209941528737545 | strengthening_recently | broad_across_symbols | retain_for_monitoring | robust_phase1_anomaly_candidate |
| opening_breakdown_5m | fwd_return_30m_benchmark_adjusted | -8.147830521920696e-05 | 0.0013312165377469042 | 9.73727073869668e-05 | 0.0001248971384484321 | 0.00010607165313558653 | 0.00014914877829141915 | 0.00025969173293560743 | strengthening_recently | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |
| opening_breakdown_5m | fwd_return_120m_benchmark_adjusted | -0.0002410536071693059 | 0.0016797594709836285 | 0.00026939198141917586 | 0.0003897176939062774 | 0.00029534270288422704 | 0.00026691274251788855 | 0.000732417858671397 | persistent | broad_across_symbols | retain_for_monitoring | robust_phase1_anomaly_candidate |
| session_range_position | fwd_return_120m_benchmark_adjusted | 0.000335616510710679 | 1.4490259059452902e-05 | 0.00037555425660684705 | 0.0005309885600581765 | 0.0006627565016970038 | 0.0006063588662073016 | 0.001148774055764079 | strengthening_recently | broad_across_symbols | retain_for_monitoring | robust_phase1_anomaly_candidate |
| range_position_2 | fwd_return_120m_benchmark_adjusted | 0.0001343856729363324 | 1.2819456722957952e-06 | 0.00016233634960372 | 0.00020519985991995782 | 0.00024497308186255395 | 0.00024715682957321405 | 0.00043688161531463265 | strengthening_recently | broad_across_symbols | retain_for_monitoring | robust_phase1_anomaly_candidate |
| market_residual_return_60 | fwd_return_eod_benchmark_adjusted | -0.00021471485524671152 | 5.654872156393795e-08 | 0.0001962286769412458 | 8.902617992134765e-05 | 5.581844015978277e-05 | 1.670844358159229e-05 | 6.478368595708162e-05 | weakening_recently | broad_across_symbols | retain_for_monitoring | robust_phase1_anomaly_candidate |
| opening_close_location_20m | fwd_return_15m_benchmark_adjusted | 5.926100129727274e-05 | 0.00038670268669186954 | 6.79769873386249e-05 | 9.615639282856137e-05 | 9.729033627081662e-05 | 9.328388841822743e-05 | 0.0001169247116195038 | strengthening_recently | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |
| market_residual_return_20 | fwd_return_30m_beta_residual | -8.558652552892454e-05 | 0.0011964639548375869 | 6.87845895299688e-05 | 6.824968295404688e-05 | 5.300951306708157e-05 | 1.550930755911395e-05 | 8.49074604047928e-06 | weakening_recently | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |
| range_position_10_z_1560 | fwd_return_120m_benchmark_adjusted | 0.00018619436013977975 | 0.0007758095273062638 | 0.00019950662681367248 | 0.0002569093485362828 | 0.00030913992668502033 | 0.0002440510579617694 | 0.00042334641329944134 | persistent | broad_across_symbols | retain_for_monitoring | robust_phase1_anomaly_candidate |
| opening_breakdown_5m | fwd_return_60m_benchmark_adjusted | -0.00014347663091029972 | 0.002555813923706597 | 0.00016016636800486594 | 0.00021724030375480652 | 0.0001640628179302439 | 0.0002180072187911719 | 0.0004339541192166507 | strengthening_recently | broad_across_symbols | retain_for_monitoring | robust_phase1_anomaly_candidate |
| market_residual_return_20 | fwd_return_5m_benchmark_adjusted | -6.390832459146623e-05 | 2.8614073717241564e-07 | 5.965402669971809e-05 | 6.683773244731128e-05 | 5.273575516184792e-05 | 1.425221034878632e-05 | 4.456598253455013e-05 | weakening_recently | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |
| upper_wick_to_range | fwd_return_30m_beta_residual | -3.838650900434004e-05 | 0.00031242085704414436 | 3.2268493669107556e-05 | 3.447102426434867e-05 | 3.180236672051251e-05 | 4.083945896127261e-05 | 1.5502626411034726e-05 | persistent | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |
| vwap_slope | fwd_return_120m_beta_residual | 0.0002087528700940311 | 0.013165228251773395 | 0.0002558301202952862 | 0.00027542043244466186 | 0.0002454978530295193 | 0.00046541134361177683 | 0.0009753846097737551 | strengthening_recently | broad_across_symbols | retain_for_monitoring | robust_phase1_anomaly_candidate |
| opening_breakout_5m | fwd_return_60m_benchmark_adjusted | 9.894405411614571e-05 | 0.00039453296829725126 | 9.02431202121079e-05 | 0.00012935596168972552 | 0.00021547010692302138 | 0.0001330900122411549 | 0.00028120080241933465 | persistent | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |
| range_position_rank_60 | fwd_return_5m_beta_residual | -5.0127931899623945e-05 | 6.285765148878392e-15 | 4.547913704300299e-05 | 2.1135965653229505e-05 | 5.607783805317013e-06 | 7.545671905972995e-06 | 3.377670145709999e-05 | weakening_recently | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |
| opening_breakout_5m | fwd_return_120m_benchmark_adjusted | 0.000209574522159528 | 0.0006459184926797259 | 0.00018707245180848986 | 0.00023958738893270493 | 0.00042094592936336994 | 0.0002807661658152938 | 0.0007159075466915965 | persistent | broad_across_symbols | retain_for_monitoring | robust_phase1_anomaly_candidate |
| return_rank_4 | fwd_return_5m_beta_residual | -4.687235923483968e-05 | 1.7663173069784217e-11 | 3.5595228837337345e-05 | 3.653743988252245e-05 | 3.0100012736511417e-05 | 1.932687700900715e-05 | 1.1108393664471805e-05 | weakening_recently | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |
| universe_breadth_positive | fwd_return_30m_benchmark_adjusted | 5.79671686864458e-05 | 9.402991439612962e-05 | 3.076868233620189e-05 | 1.907629302877467e-05 | 2.6127656383323483e-05 | 6.245445092645241e-06 | -2.464965837134514e-05 | weakening_recently | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |
| return_outlier_score_24 | fwd_return_5m_beta_residual | -3.7893090848228894e-05 | 9.519842542119354e-12 | 3.4377637348370627e-05 | 2.9919799999333918e-05 | 1.3365770428208634e-05 | 2.6475536287762225e-06 | 1.8495524273021147e-05 | weakening_recently | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |
| universe_breadth_positive | fwd_return_30m_beta_residual | 5.947201316303108e-05 | 0.00031242085704414436 | 3.77171891159378e-05 | 2.0397254047566094e-05 | 2.1533634935622104e-05 | 1.4635316802014131e-05 | -8.030028766370378e-06 | weakening_recently | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |
| opening_breakout_5m | fwd_return_30m_benchmark_adjusted | 5.051733342043008e-05 | 0.00021408020584633906 | 4.6620280045317486e-05 | 6.338105595204979e-05 | 0.00010705055319704115 | 6.589020631508902e-05 | 0.00016232117195613682 | persistent | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |
| universe_breadth_positive | fwd_return_60m_beta_residual | 5.159251122677233e-05 | 0.06507347112166485 | 2.606773705338128e-05 | 4.1462590161245316e-05 | 3.7934500142000616e-05 | 3.2375974114984274e-05 | -2.70615128101781e-05 | persistent | broad_across_symbols | retain_for_monitoring | exploratory_relationship |
| upper_wick_to_range | fwd_return_30m_benchmark_adjusted | -3.7594079913105816e-05 | 0.0007532279578292688 | 3.426965122343972e-05 | 3.523561099427752e-05 | 3.543146885931492e-05 | 3.0285422326414846e-05 | -9.98311134026153e-06 | persistent | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |
| breakdown_magnitude_5 | fwd_return_5m_beta_residual | -2.9168805212975712e-05 | 7.242277796527623e-08 | 2.419289739918895e-05 | 2.5182704121107236e-05 | 2.0695744751719758e-05 | 1.9053215510211885e-05 | 1.488825091655599e-05 | persistent | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |
| lower_wick_to_range | fwd_return_30m_benchmark_adjusted | 2.3561535044791526e-05 | 0.003460721247191744 | 2.8064432626706548e-05 | 1.2103354492865037e-05 | 1.7574197045178153e-05 | 8.910747965273913e-06 | -4.621791958925314e-06 | weakening_recently | moderately_concentrated | retain_for_monitoring | statistically_significant_discovery |
| range_position_rank_8 | fwd_return_5m_beta_residual | -4.3804682718473487e-05 | 9.519842542119354e-12 | 4.006938615930267e-05 | 4.144156991969794e-05 | 2.5779807401704602e-05 | 1.0778927389765158e-05 | 1.7488020603195764e-05 | weakening_recently | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |
| session_range_position | fwd_return_15m_benchmark_adjusted | 6.506909448944498e-05 | 7.370893216673658e-06 | 7.493007433367893e-05 | 9.752718324307352e-05 | 0.00012737450015265495 | 0.00011288287350907922 | 0.0001863081706687808 | strengthening_recently | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |
| opening_breakout_5m | fwd_return_15m_benchmark_adjusted | 3.042495700356085e-05 | 7.110628220775165e-05 | 2.815779225784354e-05 | 4.002346031484194e-05 | 6.327091250568628e-05 | 4.187662489130162e-05 | 0.00010047265095636249 | persistent | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |
| upper_wick_to_range | fwd_return_60m_beta_residual | -5.145224895386491e-05 | 0.00011174691831353724 | 3.25429646181874e-05 | 2.4614771973574534e-05 | 5.991489160805941e-06 | 1.4622868548030965e-05 | -5.118648550705984e-05 | weakening_recently | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |
| relative_volume_inclusive_4 | fwd_return_30m_beta_residual | 2.9614803679578472e-05 | 8.037736738581598e-05 | 2.1999927412252873e-05 | 1.940919719345402e-05 | 2.93006633000914e-05 | 2.1327448394004023e-06 | -8.383783097087871e-06 | weakening_recently | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |
| upper_wick_to_range | fwd_return_60m_benchmark_adjusted | -5.8391709899296984e-05 | 7.39096759521241e-06 | 4.566228744806722e-05 | 4.095729309483431e-05 | 2.069499169010669e-05 | -1.234685714734951e-05 | -7.451014243997633e-05 | historically_strong_but_currently_weak | broad_across_symbols | reject_as_concentrated_or_unstable | statistically_significant_discovery |
| consecutive_positive_bars | fwd_return_120m_benchmark_adjusted | 7.393348641926423e-05 | 0.0004554675106815489 | 7.851643749745563e-05 | 4.312236706027761e-05 | 8.069127216003835e-05 | 5.3647367167286575e-05 | 0.00013405624486040324 | persistent | moderately_concentrated | reject_as_concentrated_or_unstable | statistically_significant_discovery |
| breakdown_magnitude_1 | fwd_return_5m_beta_residual | -2.6351772703492315e-05 | 7.762174882112958e-05 | 2.4360040697501972e-05 | 2.6647547201719135e-05 | 1.618809073988814e-05 | 3.662667950266041e-06 | 4.4245607568882406e-06 | weakening_recently | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |
| relative_volume_inclusive_3 | fwd_return_15m_beta_residual | 2.5276297492382582e-05 | 0.0006824603315469074 | 1.9432347471592948e-05 | 1.4393088349606842e-05 | 1.9231141777709126e-05 | 2.068742469418794e-05 | 3.8371705159079283e-05 | persistent | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |
| opening_breakout_10m | fwd_return_30m_benchmark_adjusted | 4.277046024903086e-05 | 0.0018518681275720241 | 4.2329364077886567e-05 | 4.084213651367463e-05 | 6.298965308815241e-05 | 7.725552859483287e-05 | 0.00015099014854058623 | strengthening_recently | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |
| relative_volume_inclusive_5 | fwd_return_30m_beta_residual | 4.037725375383161e-05 | 0.00013856103451884665 | 3.257248681620695e-05 | 2.7171237888978794e-05 | 3.740586544154212e-05 | 4.07101788368891e-06 | 1.3768882126896642e-05 | weakening_recently | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |
| lower_wick_to_range | fwd_return_60m_benchmark_adjusted | 3.625852605182445e-05 | 0.005026556473715328 | 3.8260459405137226e-05 | 1.466713729314506e-05 | 4.390336471260525e-06 | -9.807179594645277e-06 | -4.386726504890248e-05 | historically_strong_but_currently_weak | moderately_concentrated | reject_as_concentrated_or_unstable | statistically_significant_discovery |
| relative_volume_inclusive_3 | fwd_return_60m_beta_residual | 3.6624693166231737e-05 | 0.002106192002363141 | 3.565334918675944e-05 | 2.327565016457811e-05 | 3.0323097234941088e-05 | -3.638523594418075e-06 | -3.203528467565775e-05 | historically_strong_but_currently_weak | broad_across_symbols | reject_as_concentrated_or_unstable | statistically_significant_discovery |
| breakdown_magnitude_6 | fwd_return_5m_beta_residual | -2.830777111739735e-05 | 1.5928707989401361e-07 | 2.3069809685694054e-05 | 2.334847886231728e-05 | 1.9185739802196622e-05 | 1.9465813238639385e-05 | 5.15184729010798e-06 | persistent | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |
| relative_volume_inclusive_2 | fwd_return_30m_beta_residual | 2.7975082048214972e-05 | 0.002522276032449144 | 2.077214594464749e-05 | 1.4624801224272233e-05 | 8.861185960995499e-06 | 2.607720716696349e-06 | -1.4032792932994198e-05 | weakening_recently | broad_across_symbols | retain_for_monitoring | statistically_significant_discovery |

## Regime summary

| feature | target | regime_summary_label |
| --- | --- | --- |
| session_range_position | fwd_return_60m_benchmark_adjusted | regime_persistent |
| session_range_position | fwd_return_30m_benchmark_adjusted | regime_persistent |
| close_location | fwd_return_120m_benchmark_adjusted | regime_persistent |
| return_consistency_5 | fwd_return_120m_benchmark_adjusted | regime_persistent |
| range_position_1 | fwd_return_120m_benchmark_adjusted | regime_persistent |
| market_residual_return_60 | fwd_return_30m_beta_residual | regime_persistent |
| range_position_10_z_4680 | fwd_return_120m_benchmark_adjusted | regime_persistent |
| market_residual_return_20 | fwd_return_5m_beta_residual | regime_persistent |
| lower_high | fwd_return_120m_benchmark_adjusted | regime_persistent |
| market_residual_return_60 | fwd_return_eod_beta_residual | regime_persistent |
| opening_close_location_20m | fwd_return_30m_benchmark_adjusted | regime_persistent |
| opening_breakdown_5m | fwd_return_30m_benchmark_adjusted | regime_persistent |
| opening_breakdown_5m | fwd_return_120m_benchmark_adjusted | regime_persistent |
| session_range_position | fwd_return_120m_benchmark_adjusted | regime_persistent |
| range_position_2 | fwd_return_120m_benchmark_adjusted | regime_persistent |
| market_residual_return_60 | fwd_return_eod_benchmark_adjusted | regime_persistent |
| opening_close_location_20m | fwd_return_15m_benchmark_adjusted | regime_persistent |
| market_residual_return_20 | fwd_return_30m_beta_residual | regime_persistent |
| range_position_10_z_1560 | fwd_return_120m_benchmark_adjusted | regime_persistent |
| opening_breakdown_5m | fwd_return_60m_benchmark_adjusted | regime_persistent |
| market_residual_return_20 | fwd_return_5m_benchmark_adjusted | regime_persistent |
| upper_wick_to_range | fwd_return_30m_beta_residual | regime_persistent |
| vwap_slope | fwd_return_120m_beta_residual | regime_persistent |
| opening_breakout_5m | fwd_return_60m_benchmark_adjusted | regime_persistent |
| range_position_rank_60 | fwd_return_5m_beta_residual | regime_persistent |
| opening_breakout_5m | fwd_return_120m_benchmark_adjusted | regime_persistent |
| return_rank_4 | fwd_return_5m_beta_residual | regime_persistent |
| universe_breadth_positive | fwd_return_30m_benchmark_adjusted | regime_persistent |
| return_outlier_score_24 | fwd_return_5m_beta_residual | regime_persistent |
| universe_breadth_positive | fwd_return_30m_beta_residual | regime_persistent |
| opening_breakout_5m | fwd_return_30m_benchmark_adjusted | regime_persistent |
| universe_breadth_positive | fwd_return_60m_beta_residual | regime_persistent |
| upper_wick_to_range | fwd_return_30m_benchmark_adjusted | regime_persistent |
| breakdown_magnitude_5 | fwd_return_5m_beta_residual | regime_persistent |
| lower_wick_to_range | fwd_return_30m_benchmark_adjusted | regime_persistent |
| range_position_rank_8 | fwd_return_5m_beta_residual | regime_persistent |
| session_range_position | fwd_return_15m_benchmark_adjusted | regime_persistent |
| opening_breakout_5m | fwd_return_15m_benchmark_adjusted | regime_persistent |
| upper_wick_to_range | fwd_return_60m_beta_residual | regime_persistent |
| relative_volume_inclusive_4 | fwd_return_30m_beta_residual | regime_persistent |
| upper_wick_to_range | fwd_return_60m_benchmark_adjusted | regime_persistent |
| consecutive_positive_bars | fwd_return_120m_benchmark_adjusted | regime_persistent |
| breakdown_magnitude_1 | fwd_return_5m_beta_residual | regime_persistent |
| relative_volume_inclusive_3 | fwd_return_15m_beta_residual | regime_persistent |
| opening_breakout_10m | fwd_return_30m_benchmark_adjusted | regime_persistent |
| relative_volume_inclusive_5 | fwd_return_30m_beta_residual | regime_persistent |
| lower_wick_to_range | fwd_return_60m_benchmark_adjusted | regime_persistent |
| relative_volume_inclusive_3 | fwd_return_60m_beta_residual | regime_persistent |
| breakdown_magnitude_6 | fwd_return_5m_beta_residual | regime_persistent |
| relative_volume_inclusive_2 | fwd_return_30m_beta_residual | regime_persistent |

## Scope summary

| feature | target | sector_scope_status | industry_scope_status | scope_classification |
| --- | --- | --- | --- | --- |
| session_range_position | fwd_return_60m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| session_range_position | fwd_return_30m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| close_location | fwd_return_120m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| return_consistency_5 | fwd_return_120m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| range_position_1 | fwd_return_120m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| market_residual_return_60 | fwd_return_30m_beta_residual | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| range_position_10_z_4680 | fwd_return_120m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| market_residual_return_20 | fwd_return_5m_beta_residual | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| lower_high | fwd_return_120m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| market_residual_return_60 | fwd_return_eod_beta_residual | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| opening_close_location_20m | fwd_return_30m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| opening_breakdown_5m | fwd_return_30m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| opening_breakdown_5m | fwd_return_120m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| session_range_position | fwd_return_120m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| range_position_2 | fwd_return_120m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| market_residual_return_60 | fwd_return_eod_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| opening_close_location_20m | fwd_return_15m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| market_residual_return_20 | fwd_return_30m_beta_residual | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| range_position_10_z_1560 | fwd_return_120m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| opening_breakdown_5m | fwd_return_60m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| market_residual_return_20 | fwd_return_5m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| upper_wick_to_range | fwd_return_30m_beta_residual | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| vwap_slope | fwd_return_120m_beta_residual | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| opening_breakout_5m | fwd_return_60m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| range_position_rank_60 | fwd_return_5m_beta_residual | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| opening_breakout_5m | fwd_return_120m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| return_rank_4 | fwd_return_5m_beta_residual | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| universe_breadth_positive | fwd_return_30m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| return_outlier_score_24 | fwd_return_5m_beta_residual | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| universe_breadth_positive | fwd_return_30m_beta_residual | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| opening_breakout_5m | fwd_return_30m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| universe_breadth_positive | fwd_return_60m_beta_residual | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| upper_wick_to_range | fwd_return_30m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| breakdown_magnitude_5 | fwd_return_5m_beta_residual | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| lower_wick_to_range | fwd_return_30m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| range_position_rank_8 | fwd_return_5m_beta_residual | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| session_range_position | fwd_return_15m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| opening_breakout_5m | fwd_return_15m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| upper_wick_to_range | fwd_return_60m_beta_residual | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| relative_volume_inclusive_4 | fwd_return_30m_beta_residual | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| upper_wick_to_range | fwd_return_60m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| consecutive_positive_bars | fwd_return_120m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| breakdown_magnitude_1 | fwd_return_5m_beta_residual | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| relative_volume_inclusive_3 | fwd_return_15m_beta_residual | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| opening_breakout_10m | fwd_return_30m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| relative_volume_inclusive_5 | fwd_return_30m_beta_residual | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| lower_wick_to_range | fwd_return_60m_benchmark_adjusted | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| relative_volume_inclusive_3 | fwd_return_60m_beta_residual | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| breakdown_magnitude_6 | fwd_return_5m_beta_residual | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |
| relative_volume_inclusive_2 | fwd_return_30m_beta_residual | unavailable_missing_point_in_time_sector_data | unavailable_missing_point_in_time_industry_data | insufficient_scope_evidence |

## Exact-time summary

| feature | target | strongest_exact_decision_time | weakest_exact_decision_time | time_concentration_label |
| --- | --- | --- | --- | --- |
| session_range_position | fwd_return_60m_benchmark_adjusted | 09:35 | 10:35 | persistent_through_session |
| session_range_position | fwd_return_30m_benchmark_adjusted | 09:35 | 15:25 | persistent_through_session |
| close_location | fwd_return_120m_benchmark_adjusted | 09:35 | 11:50 | persistent_through_session |
| return_consistency_5 | fwd_return_120m_benchmark_adjusted | 09:55 | 12:05 | persistent_through_session |
| range_position_1 | fwd_return_120m_benchmark_adjusted | 09:35 | 11:50 | persistent_through_session |
| market_residual_return_60 | fwd_return_30m_beta_residual | 14:45 | 15:10 | persistent_through_session |
| range_position_10_z_4680 | fwd_return_120m_benchmark_adjusted | 10:25 | 13:50 | persistent_through_session |
| market_residual_return_20 | fwd_return_5m_beta_residual | 15:55 | 11:20 | persistent_through_session |
| lower_high | fwd_return_120m_benchmark_adjusted | 09:40 | 10:05 | persistent_through_session |
| market_residual_return_60 | fwd_return_eod_beta_residual | 15:55 | 14:55 | persistent_through_session |
| opening_close_location_20m | fwd_return_30m_benchmark_adjusted | 09:55 | 15:25 | persistent_through_session |
| opening_breakdown_5m | fwd_return_30m_benchmark_adjusted | 09:55 | 15:20 | persistent_through_session |
| opening_breakdown_5m | fwd_return_120m_benchmark_adjusted | 09:55 | 13:45 | persistent_through_session |
| session_range_position | fwd_return_120m_benchmark_adjusted | 09:35 | 13:50 | persistent_through_session |
| range_position_2 | fwd_return_120m_benchmark_adjusted | 09:40 | 11:50 | persistent_through_session |
| market_residual_return_60 | fwd_return_eod_benchmark_adjusted | 15:55 | 15:10 | persistent_through_session |
| opening_close_location_20m | fwd_return_15m_benchmark_adjusted | 10:00 | 11:15 | persistent_through_session |
| market_residual_return_20 | fwd_return_30m_beta_residual | 11:45 | 14:20 | persistent_through_session |
| range_position_10_z_1560 | fwd_return_120m_benchmark_adjusted | 10:20 | 12:10 | persistent_through_session |
| opening_breakdown_5m | fwd_return_60m_benchmark_adjusted | 09:55 | 14:40 | persistent_through_session |
| market_residual_return_20 | fwd_return_5m_benchmark_adjusted | 15:55 | 11:20 | persistent_through_session |
| upper_wick_to_range | fwd_return_30m_beta_residual | 10:15 | 09:35 | persistent_through_session |
| vwap_slope | fwd_return_120m_beta_residual | 09:55 | 13:55 | persistent_through_session |
| opening_breakout_5m | fwd_return_60m_benchmark_adjusted | 11:25 | 09:35 | persistent_through_session |
| range_position_rank_60 | fwd_return_5m_beta_residual | 15:55 | 15:10 | persistent_through_session |
| opening_breakout_5m | fwd_return_120m_benchmark_adjusted | 10:25 | 09:35 | persistent_through_session |
| return_rank_4 | fwd_return_5m_beta_residual | 15:55 | 10:35 | persistent_through_session |
| universe_breadth_positive | fwd_return_30m_benchmark_adjusted | 09:45 | 10:00 | persistent_through_session |
| return_outlier_score_24 | fwd_return_5m_beta_residual | 15:55 | 12:55 | persistent_through_session |
| universe_breadth_positive | fwd_return_30m_beta_residual | 09:40 | 10:20 | persistent_through_session |
| opening_breakout_5m | fwd_return_30m_benchmark_adjusted | 11:35 | 15:15 | persistent_through_session |
| universe_breadth_positive | fwd_return_60m_beta_residual | 12:55 | 10:00 | persistent_through_session |
| upper_wick_to_range | fwd_return_30m_benchmark_adjusted | 09:50 | 09:35 | persistent_through_session |
| breakdown_magnitude_5 | fwd_return_5m_beta_residual | 15:55 | 10:05 | persistent_through_session |
| lower_wick_to_range | fwd_return_30m_benchmark_adjusted | 10:15 | 10:35 | persistent_through_session |
| range_position_rank_8 | fwd_return_5m_beta_residual | 15:55 | 10:35 | persistent_through_session |
| session_range_position | fwd_return_15m_benchmark_adjusted | 09:55 | 10:40 | persistent_through_session |
| opening_breakout_5m | fwd_return_15m_benchmark_adjusted | 09:40 | 14:50 | persistent_through_session |
| upper_wick_to_range | fwd_return_60m_beta_residual | 10:15 | 09:35 | persistent_through_session |
| relative_volume_inclusive_4 | fwd_return_30m_beta_residual | 10:55 | 11:00 | persistent_through_session |
| upper_wick_to_range | fwd_return_60m_benchmark_adjusted | 10:15 | 09:35 | persistent_through_session |
| consecutive_positive_bars | fwd_return_120m_benchmark_adjusted | 10:25 | 09:35 | time_unstable |
| breakdown_magnitude_1 | fwd_return_5m_beta_residual | 15:55 | 09:55 | persistent_through_session |
| relative_volume_inclusive_3 | fwd_return_15m_beta_residual | 11:45 | 09:50 | persistent_through_session |
| opening_breakout_10m | fwd_return_30m_benchmark_adjusted | 11:35 | 15:15 | persistent_through_session |
| relative_volume_inclusive_5 | fwd_return_30m_beta_residual | 13:20 | 14:55 | persistent_through_session |
| lower_wick_to_range | fwd_return_60m_benchmark_adjusted | 10:15 | 09:40 | persistent_through_session |
| relative_volume_inclusive_3 | fwd_return_60m_beta_residual | 10:15 | 09:50 | persistent_through_session |
| breakdown_magnitude_6 | fwd_return_5m_beta_residual | 15:55 | 10:05 | persistent_through_session |
| relative_volume_inclusive_2 | fwd_return_30m_beta_residual | 09:55 | 09:40 | persistent_through_session |

## Phase 2 recommendation

| feature | target | phase2_recommendation | phase2_recommendation_reason | phase2_main_limitation | phase2_suggested_test |
| --- | --- | --- | --- | --- | --- |
| session_range_position | fwd_return_60m_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| session_range_position | fwd_return_30m_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| close_location | fwd_return_120m_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| return_consistency_5 | fwd_return_120m_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| range_position_1 | fwd_return_120m_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| market_residual_return_60 | fwd_return_30m_beta_residual | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| range_position_10_z_4680 | fwd_return_120m_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| market_residual_return_20 | fwd_return_5m_beta_residual | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| lower_high | fwd_return_120m_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| market_residual_return_60 | fwd_return_eod_beta_residual | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| opening_close_location_20m | fwd_return_30m_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| opening_breakdown_5m | fwd_return_30m_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| opening_breakdown_5m | fwd_return_120m_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| session_range_position | fwd_return_120m_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| range_position_2 | fwd_return_120m_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| market_residual_return_60 | fwd_return_eod_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| opening_close_location_20m | fwd_return_15m_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| market_residual_return_20 | fwd_return_30m_beta_residual | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| range_position_10_z_1560 | fwd_return_120m_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| opening_breakdown_5m | fwd_return_60m_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| market_residual_return_20 | fwd_return_5m_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| upper_wick_to_range | fwd_return_30m_beta_residual | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| vwap_slope | fwd_return_120m_beta_residual | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| opening_breakout_5m | fwd_return_60m_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| range_position_rank_60 | fwd_return_5m_beta_residual | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| opening_breakout_5m | fwd_return_120m_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| return_rank_4 | fwd_return_5m_beta_residual | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| universe_breadth_positive | fwd_return_30m_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| return_outlier_score_24 | fwd_return_5m_beta_residual | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| universe_breadth_positive | fwd_return_30m_beta_residual | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| opening_breakout_5m | fwd_return_30m_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| universe_breadth_positive | fwd_return_60m_beta_residual | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| upper_wick_to_range | fwd_return_30m_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| breakdown_magnitude_5 | fwd_return_5m_beta_residual | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| lower_wick_to_range | fwd_return_30m_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| range_position_rank_8 | fwd_return_5m_beta_residual | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| session_range_position | fwd_return_15m_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| opening_breakout_5m | fwd_return_15m_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| upper_wick_to_range | fwd_return_60m_beta_residual | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| relative_volume_inclusive_4 | fwd_return_30m_beta_residual | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| upper_wick_to_range | fwd_return_60m_benchmark_adjusted | reject_as_concentrated_or_unstable | Effect is unstable across current descriptive slices | Aggregate significance is not operationally stable | No immediate strategy test; monitor for stable re-emergence |
| consecutive_positive_bars | fwd_return_120m_benchmark_adjusted | reject_as_concentrated_or_unstable | Effect is unstable across current descriptive slices | Aggregate significance is not operationally stable | No immediate strategy test; monitor for stable re-emergence |
| breakdown_magnitude_1 | fwd_return_5m_beta_residual | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| relative_volume_inclusive_3 | fwd_return_15m_beta_residual | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| opening_breakout_10m | fwd_return_30m_benchmark_adjusted | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| relative_volume_inclusive_5 | fwd_return_30m_beta_residual | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| lower_wick_to_range | fwd_return_60m_benchmark_adjusted | reject_as_concentrated_or_unstable | Effect is unstable across current descriptive slices | Aggregate significance is not operationally stable | No immediate strategy test; monitor for stable re-emergence |
| relative_volume_inclusive_3 | fwd_return_60m_beta_residual | reject_as_concentrated_or_unstable | Effect is unstable across current descriptive slices | Aggregate significance is not operationally stable | No immediate strategy test; monitor for stable re-emergence |
| breakdown_magnitude_6 | fwd_return_5m_beta_residual | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
| relative_volume_inclusive_2 | fwd_return_30m_beta_residual | retain_for_monitoring | Evidence does not yet support a specific Phase 2 strategy scope | Insufficient descriptive breadth, scope, or timing evidence | Refresh diagnostics after additional predeclared data or metadata |
