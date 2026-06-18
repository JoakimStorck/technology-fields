# Startinstruktion: Papper 3 — teknologikalibrering och regimkörningar

Vi arbetar på svenska; manuskriptet är på engelska. Detta är GTS-modellpapperet
i forskningsprogrammet The Geometry of Work (medförfattare Jonatan Andersson).
Positionering: **teoretiskt bidrag med illustrativ empirisk tillämpning**,
fälttidskrift (Labour Economics / JoLE). Titeln parallelliserar avsiktligt
Acemoglu–Restrepo 2019. Empirisk ambition hör till Paper 4, inte hit.

## Syftet med denna fas — läs först

AI-exponering är **en instans** av en Technology — ett fält phi_K(r) över
taskdisken bland många. Poängen är INTE att mäta AI-exponering noggrant; det är
att visa att fältmodellen kan ta emot vilken teknologi som helst som ett
phi_K-fält och att mekanismen (övertagande på prismarginalen, reinstatement på
gradientringen, deficitgaten) producerar fördelningskonsekvenser. AI används som
illustration för att den är konkret och samtida. Den andra halvan av bidraget är
att visa fältmodellens **generalitet**: att samma maskineri beskriver
routinisering (Autor 2003, Technical-Physical-polen), robotar, generativ AI osv.
som kontrasterande phi_K. Generaliteten bärs av teorin och av kontrasterande
fall i §5, inte av mer empiri.

Konsekvens: exponeringskällan ska vara tillräckligt bra för en illustration, inte
optimal. Jaga inte precision i AI-fältet — det vore samma överreach mot empirin
som den skrotade DiD:n. (Yin et al. 2026 visar att enskilda LLM-baserade
exponeringsmått varierar artonfaldigt mellan modeller; det är ett ARGUMENT för
ramen — phi_K är en generell storhet, vilken källa som helst är en realisering —
inte ett problem att lösa.)

## Var vi står

- Modellsektionen (§3) och kalibreringstexten (§4) är klara och pushade, med
  planpostulatet, prisfunktionalen, q_k-fälten, gaten över prissatta kluster.
- Inledningen (§1/abstract) är PAUSAD med avsikt: den påstår fortfarande en
  skrotad DiD (ChatGPT-lansering, 741 yrken, "six specifications", "clean
  pre-trends"). Den skrivs om EFTER att beräkningarna är klara, eftersom introt
  inte kan beskriva empiriken ärligt förrän den finns.
- §4.7 (Technology calibration), §5 (Illustrative Cases), §6 (Empirical) är
  `\todo`-skal.
- **Detta arbete:** bygg exponeringsfältet, kalibrera teknologin, kör regimerna,
  fyll §4.7/§5/§6.

## Repo och arbetsflöde

- Repo (klona färskt): `https://github.com/JoakimStorck/technology-fields`
  Innehåller `model/` (data, price_field, technology, capability_field),
  skript 00–06, `scripts/run_all.py`, `scripts/07_build_exposure.py` (se nedan),
  `data/` med MANIFEST, `results/`, `paper/`.
- Paper 1-repo: `https://github.com/JoakimStorck/geometry-of-work` (referens).
- Joakim har inget annat publicerat repo. All ny kod bor i technology-fields.
- Arbetssätt: du klonar, arbetar lokalt, levererar enskilda nya/ändrade filer
  (inte zip) via outputs; Joakim committar. Verifiera varje kvantitativt
  påstående numeriskt innan det skrivs i manus — inga "ligger inom bandet" utan
  körning. Vid osäkerhet: fråga, kör inte vidare på antaganden. Direkt, kritisk
  granskning framför medhåll. Lättviktig kod utan onödiga extramoduler.
- Skripten ska vara begripliga för någon som möter dem kalla i repot:
  inga referenser till vår dialog eller till interna "faser/steg" i kod, namn
  eller kommentarer.

## Etablerat — ompröva inte utan att flagga explicit

1. **Lönemodellen** = fältet ln Π = m0 + m1·cosξ + m2·sinξ + χ(m3 + m4·cosξ
   + m5·sinξ), skattat i Paper 1 (N=785, HC3), replikerat exakt i 01/02.
2. **Planpostulatet:** kapabilitetsinnehållet är till första ordningen ett
   2D-plan; taskdisken är dess polarkarta (χ amplitud, ξ fas). Paper 1:s fyra
   regulariteter är konsekvenser. q_k och Π ärver form ur planet. Högre
   harmoniker = textur (mätt en gång, av i teorin, på i känslighet).
3. **Gaten över prissatta kluster:** δ_o = Σ_{k∈{S1,S2}} v_k·max(q_k−q_{o,k},0),
   v_k = prisfunktionalens komponenter (S1 +0.328, S2 +0.049). A1/A2 ligger i
   kärnan och gatar inte. All-kluster-varianten = känslighet.
4. **Skiktad arkitektur:** teorilagret familjefritt. Familjekilen η_g är MÄTT
   (05), hålls fast, simuleringar med/utan kil, η omoptimeras aldrig. Den
   institutionella valideringen av kilen är INTE gjord — påstå den inte.
5. **Buntoperatorn:** b_post onormaliserad; w_o är prissatt innehåll. (1−a)
   ingår exakt en gång (i h_o). Massunderskott D_o − B_o/L_o = arbetarlagrets
   omsorteringsinput. Tre mått: h_o, k_o, h_o+k_o.
6. **Giltighetsdomän:** fältet skattat på χ ≤ 0.752; tasks når χ = 1. Vi klipper
   inte; extrapolation är explicit åtagande med känslighetsnot.
7. **Paper 2 citeras inte.** d(o,o′) = ‖μ_o − μ_{o′}‖ självständigt via
   centroidavstånd + Gathmann–Schönberg.
8. **Pinning-policy:** under utveckling kör 00 mot SENASTE geometry-of-work.
   Data låses först när Paper 1 är accepterat. MANIFEST loggar vad vi körde mot.

## Exponeringsfältet — källa och pipeline (LÖST)

**Källa:** Gmyrek et al. 2025, ILO Working Paper 140, på Pawel Gmyreks publika
GitHub: `https://github.com/pgmyrek/POLAND_2025_GenAI_scores_6digit_occupations`.
Filen `POLAND_FINAL_6digit_scores.xlsx` (29 700 rader, 28 790 unika engelska
task-texter, kolumn `score25` i [0.05, 0.80], samt `task_full_text_ENG`). Detta
är 2025-efterföljaren till den data Joakim redan använt; verifierad mot filen.
Repot har också `mean6d_pl`/`SD6d_pl` (Gmyreks egna μ/σ per yrke — möjlig
validering av μ–σ-realiseringen) och `ISCO_mapping.csv`.

**Avgörande encoder-krav:** geometrin är byggd med OpenAI text-embedding-3-large
(d=3072). phi(r) MÅSTE byggas i samma rum — annars får exponeringsytan fel form
och teknologin kalibreras mot fel måltavla. Därför kräver byggsteget OpenAI-API
+ nyckel och körs på **Joakims maskin (Pop!_OS)**, inte i sandlådan. Allt
nedströms (frysning, kalibrering, regimer) är encoderoberoende.

**Skriptet finns redan:** `scripts/07_build_exposure.py` (lättviktigt,
fristående, inga projektinterna moduler; kNN+isoton enbart — ridge/ensemble
borttagna; INGEN rescaling — isoton kalibrering ensam; OOD-similaritet som
diagnostik, ingen filtrering). Allt utom OpenAI-anropet är testat. Joakim kör:
```
python scripts/07_build_exposure.py \
    --gmyrek /sökväg/POLAND_FINAL_6digit_scores.xlsx \
    --onet-tasks /sökväg/"Task Statements.xlsx"
```
Det skriver `data/onet_task_exposure.csv` (onet_code, task_id, phi,
train_similarity). Embeddings cachas lokalt (.npy) så en omkörning inte ringer
API:et igen.

## Begreppsmappning (kritisk — håll isär)

- **phi(r) = observerad AI-exponering per task** (från 07). Indata.
- **phi_K(r) = teknologins postulerade effektivitetsfält** (gaussian, eq. phi-K),
  parametrar (p_K, z_K, A_K). Detta KALIBRERAS.
- Kalibrering = hitta (p_K, z_K, A_K) så att phi_K bäst återger den observerade
  phi(r)-ytan över disken. phi(r) är måltavlan.

## μ–σ-spridningen — om den används, så på rätt grund

σ_φ inom ett yrke är INTE ett importerat Brynjolfsson-mått utan exakt vad
buntoperatorn formaliserar: spridningen i a(r) över bunten b_o. §7 säger redan
att displacement sker på den inre marginalen. Presenteras som mätbar realisering
av en mekanism modellen har, inte som extern deskription. Gmyreks `SD6d_pl` kan
tjäna som jämförelse. Riskklassningen (Gradient 1–4) från ILO-rapporten ANVÄNDS
INTE — den var Gmyreks inkluderingslogik, inte vår.

## Arbetsordning — invänta OK mellan stegen

- **A. KALIBRERINGSSPECIFIKATION (gör detta först, ingen kod).** Definiera
  förlustfunktionen för (p_K, z_K, A_K) mot phi(r)-ytan exakt. Avgör s_K
  (replacing/augmenting) och attachment-skalan ℓ. Avgör också omfånget: ska §6
  ha ett eget empiriskt test (centroidskifte Δμ_o·∇Π mot observerade
  löneförändringar) eller = kalibrering + regimillustration? (Lutning:
  kalibrering + illustration + μ–σ-realisering; centroidtestet på sin höjd som
  ärligt icke-kausal konsistensfigur.) Enheter: R, Π i USD/h; A_K dimensionslös;
  τ marginbredd i samma enhet som s_K·phi_K − R/Π.
- **B. KALIBRERINGSSKRIPT** (`08_calibrate_technology.py`, kör var som helst):
  fryser `onet_task_exposure.csv` med proveniens i MANIFEST (extern tredje källa,
  inte Paper 1-export), kalibrerar phi_K mot phi(r). `run_all.py` uppdateras (07
  ingår INTE i run_all — det kräver API/Joakims maskin; dokumentera det).
- **C. REGIMKÖRNINGAR:** kalibrerad teknologi, med/utan kil. Tröskeln i z_K som
  STRUKTURELLT resultat (inte kausal skattning). μ–σ-realiseringen.
- **D. GENERALITET (§5):** kontrasterande phi_K (routinisering, robotar, GenAI)
  som visar att maskinen är teknologi-agnostisk.
- **E. TEXT:** fyll §4.7, §5, §6 på körda siffror; varje tal pekbart till en
  skriptoutput. Därefter kan inledningen (§1/abstract) skrivas om och
  DiD-påståendena tas bort.

## Börja med A.
Ställ kalibreringsspecifikationen och omfångsfrågan innan någon kod skrivs.
Notera att 07 redan finns och att phi(r) byggs på Joakims maskin.
