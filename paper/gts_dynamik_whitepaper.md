# Den dynamiska GTS-modellen

**Antaganden, matematik och systemdynamisk formulering — arbetsdokument**

Joakim Storck & Jonatan Andersson · The Geometry of Work · utkast för fortsatt arbete · reviderad 21 juni 2026

---

## 0. Status och avgränsning

Detta är ett internt arbetsdokument, inte ett pappersutkast. Det dokumenterar den dynamiska utvidgningen av den statiska GTS-modellen i Paper 3, så som vi resonerat oss fram till den. Tre ramvillkor gäller genomgående:

1. **Den statiska grunden behålls.** Dynamiken är ett lager ovanpå statiken. Där ett val rör grunden är det uttryckligen flaggat nedan; inget sådant val har gjorts tyst.
2. **Minsta tillägg som ger den dynamik vi vill se.** Genomgången av varje mekanism har drivits mot att återanvända befintliga objekt snarare än att bygga nya. Resultatet: det enda genuint nya strukturobjektet är att den obundna massan blir ett verkligt lager.
3. **Teoretisk elegans förankrad i Paper 1 är ledstjärnan.** Härledda konsekvenser ska vara satser, inte antaganden.

Ingen fråga om hur stoffet ska organiseras i papper avgörs här.

**Revisionsnot (21 juni 2026).** Bindningslagen (§5.3) är omskriven: den tidigare kapacitetsformen $\Phi(C)\,U$ är ersatt av en match-allokerad, storleks-taktbegränsad bindning (match avgör destinationen, storlek takten). Kalenderförankring tillagd (§5.2). Ny observabel §9.7: tempokvoten avgör destinationen, inte bara transienten. Ny grundfråga Ö3 (survival-grinden: absolut kostnad mot komparativ fördel). A12 nyanserad för den tvåsidiga bindnings-grinden. §10 uppdaterad för utfört arbete.

---

## 1. Den konceptuella drivkraften

Vid varje tidpunkt finns mogna teknologier, vars arbetsinnehåll är i jämvikt, och omogna, vars arbetsinnehåll är i förändring. Ekonomin är aldrig i statisk jämvikt utan ständigt under omformning. Det observerade tillståndet är en stillbild, formad av historiska förlopp.

**Två modeller, två frågor.** Statik och dynamik är inte samma modell sedd på två sätt; de är olika modeller som svarar på olika frågor.

- Den **statiska modellen beräknar en jämvikt.** Den frågar: givet teknologin, vilket tillstånd uppfyller $L = F(L)$? Svaret nås *utan mellanliggande tid*. Den dämpade Picard-iterationen i `equilibrium.solve` är en lösningsmetod, utbytbar mot vilken annan fixpunktslösare som helst; dess mellansteg är numeriska artefakter och dämpningen $\lambda$ är en konvergensparameter, inte en hastighet. Att mellanstegen *liknar* rörelse är en ytlig formlikhet utan fysikalisk innebörd.
- Den **dynamiska modellen beräknar ingen jämvikt.** Den ställer upp rörelselagar — differentialekvationer i tid (§5) — och låter tillståndet röra sig enligt dem, med tröghet och faktiskt tidsförlopp. Att systemet eventuellt *närmar sig* en jämvikt är en konsekvens av lagarna, inte något som beräknats fram, och i allmänhet nås den aldrig: målet förskjuts medan tekniken mognar.

De är besläktade men inte identiska: de dynamiska lagarna är byggda så att deras stationärpunkter ($\dot L = 0$) uppfyller ett villkor som *liknar* statikens fixpunkt. De två sammanfaller bara om den dynamiska rörelselagens stationärpunkt råkar lösa samma villkor som den statiska beräkningen — vilket den gör under fryst origo men i allmänhet inte under medflyttande origo (§7, A1). Det är därför den statiska jämvikten och den dynamiska vilopunkten är vilopunkter åt *olika slags objekt* — en lösning på ett ekvationssystem respektive en nollpunkt för ett differentialekvationshögerled — och därför kan ligga isär.

Den statiska jämvikten får därför två roller i förhållande till dynamiken:

- **Startbild** — det läge filmen börjar i (före teknologin).
- **Referensmått** — fräsch-allokeringen, dit en friktionsfri ekonomi skulle nå om alla valde om från noll.

Den dynamiska modellens *vilopunkt* — dit ekonomin faktiskt kryper när den måste släpa med sig sin nuvarande fördelning — är ett annat objekt, beräknat på ett annat sätt (integration av en rörelselag, inte lösning av ett fixpunktsvillkor), och i allmänhet en annan punkt. Gapet mellan den statiska jämvikten och den dynamiska vilopunkten mäter hur hårt historien binder ekonomin i ett spår. **Detta rör grunden** (se §7, A1).

---

## 2. Epistemiska lager

Tre lager med olika kunskapsstatus, oförändrade från den statiska grunden och utsträckta till de nya objekten:

- **Teori.** Förmågeplanet (rang-2-postulatet) och prisfunktionalen.
- **Mätning.** Frysta Paper 1-koefficienter; task radius och mobilitetsskala från Paper 1/2.
- **Simulering.** De fria primitiverna (teknologi och ekonomi) *och* alla hypotetiska objekt: nyskapade uppgifter och nyfödda yrken.

**Empirisk/hypotetisk-flagga.** Empiriska uppgifter (O*NET) och hypotetiska (modellsådda) lever fritt i samma geometri — fälten skiljer inte på dem, för modellen läser bara *position*, aldrig innehåll. Men de har olika status: de empiriska är mätningar, de hypotetiska är modellens egna förutsägelser, placerade av såningsregeln. Varje punkt och varje yrke bär en flagga om det är mätt eller fött, så att vi alltid kan köra resultatet med och utan de hypotetiska objekten och se hur mycket av dynamiken som drivs av data mot av antagande.

---

## 3. Den statiska grunden (sammanfattning)

Task layer, fält över enhetsskivan ($\xi$ = riktning/domän, $\chi$ = djup/specificitet):

$$\ln \Pi(\mathbf r) = m_0 + m_1\cos\xi + m_2\sin\xi + \chi\,(m_3 + m_4\cos\xi + m_5\sin\xi)$$

$$\varphi_K(\mathbf r) = A_K\,\exp\!\Big(-\tfrac12\big(\lVert \mathbf r - \mathbf p_K\rVert / z_K\big)^2\Big)$$

Opererad andel (kapital tar det dyraste arbetet först; löne-wedge enbart här):

$$a(\mathbf r) = \sigma\!\Big(\big(s_K\varphi_K - R/\Pi\big)/\tau\Big)$$

Förträngning per yrke och aggregat:

$$D_o = \int b_o\,a,\qquad \Delta\Gamma^D = \sum_o L_o D_o,\qquad M = \gamma\,\Delta\Gamma^D$$

Reinstatement (bundle-operatorn). Frön sås på gradientringen $\hat g = \lVert\nabla\varphi_K\rVert / \int\lVert\nabla\varphi_K\rVert$; deficit-grind ger readiness; överlevnadsgrind $(1-a)$:

$$\delta_o(\mathbf r) = \sum_{k\in K} v_k\,\max\big(q_k(\mathbf r) - q_{o,k},\,0\big),\quad e_o = e^{-\delta_o/\ell},\quad C = \sum_o L_o e_o,\quad \Phi(C) = \frac{C}{1+C}$$

$$s = M\hat g,\qquad \iota_o = s\,(1-a)\,\Phi\,\frac{L_o e_o}{C},\qquad u = s\,(1-a)\,(1-\Phi)$$

Bundle-operator (massa bevaras *inte*; massdeficiten driver omsorteringen):

$$b_o^{\text{post}} = b_o(1-a) + \iota_o/L_o$$

Efterfrågekanal (closure 1, intäktsmultiplikator), två-faktor-densitet, yrkesvärde, arbetsandel:

$$D(\mathbf r) = \big(c/\Pi\big)^{1-\eta},\quad c/\Pi = (1-a) + a\,\frac{R}{s_K\varphi_K\Pi}$$

$$n(\mathbf r) = \sum_o L_o b_o + \sum_o \iota_o,\qquad W_o = \beta\!\int h_o\,D\,\Pi\,n^{\beta-1},\qquad \Lambda = \frac{\int D\,\Pi\,H\,n^{\beta-1}}{\int D\,\Pi\,n^{\beta}}$$

Worker layer (reading A, origin-destination-logit, frysta för-teknologi-centroider):

$$L_o = \sum_{o'} L_{o'}^0\,P(o\mid o'),\quad P(o\mid o') = \operatorname{softmax}_o\!\Big(\frac{W_o - c\,d_{oo'}}{\kappa}\Big),\quad d_{oo'} = \lVert\mu_o^0 - \mu_{o'}^0\rVert$$

Task radius (Paper 1/2), buntens bredd:

$$R_o = \sqrt{\sum_{t\in\mathcal T_o} \tilde w_{o,t}\,\lVert \mathbf x_t - \mu_o\rVert^2}$$

---

## 4. Systemdynamisk formulering

### 4.1 Föreningsprincipen

System dynamics ger **grammatiken**: lager ändras bara genom flöden (bevarande), flöden bildar slingor (återkoppling). Geometrin ger **konstitutivlagarna**: varje flödestakt är en funktional av de geometriska fälten ($\Pi$, $\varphi_K$, $e$, $d$). Geometrin är alltså inte ett eget lager utan *innehållet* i de rate-ekvationer SD lämnar tomma. Det är därför geometrin tråder genom varje flöde men aldrig bor i en egen låda.

### 4.2 Lagren

Ordet "lager" rymmer tre olika matematiska objekt, och geometrin sitter olika i vart och ett:

| Lager | Objekt | Geometrins roll |
|---|---|---|
| $A_K^{(i)}(t)$ — teknologins mognad (per fält $i$) | skalär | ingen |
| $L_o(t)$ — population över yrken | punktmått på yrkescentroider | metrik (avstånd = kostnad) |
| $U(\mathbf r,t)$ — obundet (nyskapat, ej bundet) arbete | **fält över skivan** | medium (badkar per plats) |
| $q_{o,k}(t),\ \mu_o(t)$ — yrkenas karaktär | koordinater i geometrin | tillstånd (geometrin deformeras) |

$U$ är det **enda genuint nya strukturobjektet**. Det ger den *obundna sådd-massan* ett hem — den tredje öden av den nyskapade massan på ringen, som statiken skapar och kastar vid varje anrop ("unbound mass performs no work and does not enter"). 

**Två konserveringsklasser, inte en.** Det är viktigt att inte överdriva vad $U$ sluter. I statiken finns två skilda massrörelser:

1. *Förträngning.* $D_o = \int b_o a$ flyttar en *uppgift* från människa till kapital inom $n$. Det arbetaren förlorar är *tid*, och den tiden är redan sluten via worker-lagret — hon bär den med sig till andra yrken genom omsorteringen. Inget här är obundet eller väntar på ett hem.
2. *Sådd.* $M = \gamma\Delta\Gamma^D$ är **ny** massa på ringen med tre öden: kapitalfångad $s\,a$, bunden $s(1-a)\Phi$, obunden $s(1-a)(1-\Phi)$. Det är den obundna delen som saknar bärare i statiken.

$U$ ger den obundna **sådd**-massan ett lager — det sluter *sådd*-liggaren, inte förträngnings-liggaren. (En tidigare version av detta dokument knöt felaktigt $U$ till massdeficiten $D_o - B_o/L_o$; den deficiten är frigjord *tid* som redan omsorteras, ett annat objekt än obunden sådd-massa.)

Konserveringsutsagan är därför tudelad och måste hållas isär:

- **Populationen bevaras.** $\sum_o \dot L_o = 0$ (§5.1). Människor skapas eller förstörs inte.
- **Uppgiftsmassan har en källa.** Sådden *skapar* massa (nya uppgifter ur intet — det är meningen). Den skapade massan förflyttas sedan konservativt mellan tre tillstånd: obunden ($U$), bunden ($b_o$ via operatorn), kapitalfångad ($K$). Inom uppgiftsmassan, *efter* sådden, bevaras massa; sådden själv är en källa.

Det $U$ ger är alltså inte en global massbevaring (det vore fel — sådden är en genuin källa) utan ett hem åt den enda av såd-massans tre öden som statiken inte bokförde. Det är den verkliga och eleganta vinsten, korrekt formulerad.

### 4.3 Den dynamiska liggaren

| Massrörelse | Typ | Verkan |
|---|---|---|
| Förträngning $h \to k$ | bevarande (inom $n$) | uppgift byter bärare människa→kapital; frigjord *tid* omsorteras via $L$ |
| Omsortering | bevarande ($\sum\dot L_o=0$) | flyttar population mellan yrken |
| **Sådd** | **källa** | skapar ny uppgiftsmassa $M$ på ringen |
| Bindning $U \to b_o$ | bevarande (inom uppgiftsmassan) | obunden massa blir bunden, skrivs in i bunten |
| Kapitalfångst $s\,a$ | bevarande (inom uppgiftsmassan) | del av sådd-massan fångas av kapital ($K$) |

Lager: $L$ (population), $b_o$:s mänskliga innehåll, $U$ (obunden sådd-massa), implicit $K$ (kapitalfångat). Den enda källan är sådden; allt annat är konservativa förflyttningar. Det är denna liggare pucklen (§9.1) vilar på.

### 4.4 Flödena och slingorna

- **Mognad** fyller $A_K$. Förstärkande slinga **R** (användning → styrka) balanserad av **B1** (mättnad).
- **Förträngning** $H \to K$ med takt $a(\mathbf r)$, gated av prisfält och teknikfält.
- **Sådd** matar $U$ — nya beständiga uppgifter, i takt med hur mycket som *nyss* förträngts (se §5, A4).
- **Bindning** tömmer $U$ — löst arbete blir någons jobb när kapacitet finns. Balanserande slinga **B2** (gapet sluts, med fördröjning).
- **Förvärv / inre glidning** rör det långsamma $q,\mu$. Balanserande slinga **B3** (decennier).
- **Trängsel**: hög $L$ på en plats sänker platsvärdet. Balanserande slinga **B4** (finns redan i $n^{\beta-1}$).

---

## 5. Rörelselagarna

### 5.1 Population (bevarad, första ordningen, medflyttande origo)

Målet är den befintliga statiska omsorteringen, men origo är **nuvarande** $L(t)$ (medflyttande), inte en frusen baslinje:

$$\dot L_o = \frac{1}{\theta_L}\Big(T_o\big(L,t\big) - L_o\Big),\qquad T_o(L,t) = \sum_{o'} L_{o'}(t)\,P\big(o\mid o';\,W(L,t)\big)$$

Eftersom $P$ är radstokastisk gäller $\sum_o \dot L_o = 0$ — **populationen är exakt bevarad**. $W(L,t)$ levereras av task layer vid nuvarande tillstånd och fält. Tidskonstanten $\theta_L$ sätts av $(c,\kappa)$ i tolkbara enheter. Den **ersätter** Picard-dämpningen $\lambda$ — den ärver inget från den. Som §1 slår fast bar $\lambda$ ingen fysik (den var en konvergensknapp); den dynamiska modellen omtolkar den alltså inte utan inför en genuin, kalibrerad hastighet $\theta_L$ i dess ställe. **Första ordningen** valdes medvetet: den ger gap och fördröjning men ingen eftersväng. Andra ordningen införs bara om data visar cykler i omställning (§7, A3).

### 5.2 Mognad (minimal, exogen)

$$\dot A_K^{(i)} = \rho_i\,A_K^{(i)}\big(1 - A_K^{(i)}/\bar A_i\big)$$

Logistisk: förstärkande tidigt (R), mättande sent (B1). Endogen variant ($\dot A_K \propto$ opererad massa) är ett tyngre, senare alternativ.

I kalenderförankrad form tolkas $t$ i *år* och logistiken förankras i en chockvaraktighet $T_{\text{shock}}$: lutningen sätts så att $A_K$ reser sig $5\%\to 95\%$ över $T_{\text{shock}}$ år (illustrativt $T_{\text{shock}}=5$), centrerad vid $T_{\text{shock}}/2$. Därmed blir $\theta_L$ och $\theta_{\text{abs}}$ kalendertidsskalor (§8), och utfallet styrs av *kvoten* mellan chocktempo $T_{\text{shock}}$ och omfördelningstempo $\theta$ (§9.7).

### 5.3 Obundet arbete (det nya lagret)

$$\partial_t U(\mathbf r,t) = \dot s(\mathbf r,t) - \sum_o \iota_o(\mathbf r,t)$$

$$\dot s(\mathbf r,t) = \gamma\,\big[\partial_t\,\Delta\Gamma^D\big]_{+}\,\hat g(\mathbf r)\,(1-a(\mathbf r,t))$$

Sådden följer *förändringstakten* i förträngningen (positiv del): det skapas nya uppgifter medan tekniken mognar, och sådden upphör när förträngningen planat ut.

**Bindningen: match allokerar, storlek begränsar takten.** Bindningsflödet $\iota_o$ — som ersätter den tidigare kapacitetsformen $\lambda_b\Phi(C)U$ — vilar på en åtskillnad som visade sig bärande: *vart* den obundna massan går avgörs av match, *hur snabbt* den binds av storlek. Vid varje plats attraheras massan till bäst match via en skärpt andel, och varje yrke binder upp till ett storleksbestämt tak:

$$t_o(\mathbf r) = U(\mathbf r)\,\frac{\mathrm{FIT}_o(\mathbf r)^{\beta}}{\sum_{o'}\mathrm{FIT}_{o'}(\mathbf r)^{\beta}},\qquad \mathrm{FIT}_o(\mathbf r) = e_o^{\text{match}}(\mathbf r)\;e^{-\lVert\mathbf r-\mu_o\rVert/\rho}$$

$$c_o = M_o/\theta_{\text{abs}},\quad M_o = b_o^{\text{orig}} + r_o,\qquad \iota_o(\mathbf r) = t_o(\mathbf r)\,\min\!\Big(1,\;\frac{c_o}{\int t_o\,d\mathbf r}\Big)$$

Anspråket $t_o$ fördelar den obundna massan efter *skärpt* match ($\beta>1$ låter bästa match dominera per plats); taket $c_o$ skalar med yrkets uppgiftsmassa $M_o$ (som växer endogent med det bundna $r_o$). Det ett mättat yrke inte hinner binda — andelen $1-\min(1,c_o/\!\int t_o)$ — återgår till $U$ och erbjuds nästa steg, då *nästa-bästa* match exponeras. Stora yrken binder därmed snabbare i absoluta tal men inte snabbare *relativt* sin storlek; residualen kaskaderar nedåt i match.

$e_o^{\text{match}}$ är ett *tvåsidigt* matchavstånd (straffar både under- och överkvalifikation), till skillnad från den statiska ensidiga readiness-grinden $e_o$ (§3), som behålls för den statiska kapaciteten. Skälet: *kunna* (statiskt hölje, överkvalifikation gratis) och *bli hem för* (dynamisk bindning) är olika frågor — ett yrke långt över en uppgifts nivå kan utföra den men ska inte dra den till sig som sitt arbete. Detta nyanserar A12 och flaggas där.

**Varför formen byttes.** Både den statiska kapacitetsandelen $\Phi(C)\,L_o e_o/C$ och en mellanliggande massviktad form $M_o\mathrm{FIT}_o/\!\sum_{o'} M_{o'}\mathrm{FIT}_{o'}$ lät storleken *multiplicera* matchen: den *absoluta* absorptionen korrelerade $\approx 0.99$ med yrkesstorlek och endast $\approx 0.04$ med matchkvalitet — storleken avgjorde destinationen, matchen bara den relativa tillväxten. Match-allokeringen med storlekstak bryter detta (korrelation mot storlek $\approx 0.04$): destinationen styrs av match, storleken degraderad från att bestämma *vart* arbetet går till *hur fort*. $\theta_{\text{abs}}$ och $\beta$ är kalibreringsrattar, inte världsbilder; deras ekonomiska innebörd avgörs av svep, inte av modellören.

**Pucklen sitter i obundenhet, inte i existens.** Uppgifter är beständiga; det är väntan på en bärare som öppnar sig under mognad ($\dot s>0$) och sluts efteråt ($\dot s\to 0$, $U$ avklingar via bindning). Uppgifters "död" är ingen egen mekanism utan en **avläsning**: en plats vars värde gått till noll (mättnad via $\eta$, eller nästa vågs $a(\mathbf r)$) bär ingen arbetskraft, vare sig uppgiften står kvar i registret eller ej.

### 5.4 Yrkenas karaktär (långsam)

$$\dot q_{o,k} = \frac{1}{\theta_q}\big(q^{\*}_{o,k}(t) - q_{o,k}\big),\qquad \theta_q \sim \text{decennier}$$

Två drivkrafter mot målnivån $q^\*$: **förvärv** (där reinstatement lönar sig sluter yrket sin deficit) och **augmenteringens inre glidning** (§6). $\mu_o(t)$ följer av buntens tyngdpunkt. *Detta är den minst specificerade rörelselagen* och bör hårdnas i fortsatt arbete.

### 5.5 Bundle-operatorn som flöde

Den statiska operatorn är integralen av en kontinuerlig omskrivning:

$$\dot b_o(\mathbf r) = -\,b_o(\mathbf r)\,\partial_t a(\mathbf r,t)\;+\;\iota_o(\mathbf r,t)/L_o$$

(Bindningsflödet $\iota_o$ är här den match-allokerade, storleks-taktbegränsade formen från §5.3.)

Bundna reinstaterade uppgifter ackumuleras i $b_o$ och blir därmed beständiga — de lever vidare i $n$, flyttar med yrket, och kan tas av nästa våg via $a(\mathbf r)$ eller dräneras via platsvärdet. Persistensen kräver alltså ingen separat beståndsbokföring; den bärs av att operatorn skriver om bunten. Detta ersätter statikens omräkning-vid-varje-anrop med ackumulation och är, vid sidan av $U$, den andra platsen där dynamiken faktiskt skiljer sig från koden i dag.

---

## 6. Augmentering som experimentratt

Karaktärsglidningen ersätta ↔ förstärka är **endogen och gratis på ersättningssidan**: den faller ut av prisgrinden $s_K\varphi_K \gtrless R/\Pi$ när $A_K$ växer. Ett ungt svagt fält når inte över grinden någonstans (förstärker bara); ett moget starkt korsar den i de dyraste riktningarna (ersätter där) men inte i de billigare. Ingen egen rörelselag behövs för skiftet.

Den frigjorda tiden när ett verktyg gör halva uppgiften är en **öppen empirisk fråga**, testad genom experiment via en andel $\alpha$:

- $\alpha = 0$: frigjord tid blir mer produktion → kostnadssänkning → kanaliseras genom $\eta$. **Ren produktivitet, redan byggd.**
- $\alpha = 1$: frigjord tid går till yrkets *andra* uppgifter → buntens tyngdpunkt glider mot det verktyget inte tog → matar $q,\mu$ (§5.4). **Inre glidning.**
- $0<\alpha<1$: blandning.

$\alpha$ är en ratt, inte en världsbild; svepet, inte modellören, ger svaret. **Varning:** den inre glidningen verkar på den långsammaste tidsskalan ($\theta_q$), så dess effekt är trög — kort sikt visar lite, lång sikt mycket. Detta är den enda kvarvarande mekanismen som ännu inte är specificerad i form (jfr §5.4).

---

## 7. Avgjorda antaganden

**A1 — Medflyttande origo / två jämvikter.** *Rör grunden.* Omsorteringen utgår från nuvarande fördelning, inte en frusen baslinje. Den statiska jämvikten är referens och startbild; dynamikens vilopunkt är utfall. Gapet mellan dem är ett resultat, inte ett fel. (Implicit i valet av system dynamics.)

**A2 — Populationen är historielös.** Den nuvarande fördelningen *är* dess minne; inga individbanor. Övergångar beror bara på var man står nu. Ett långsammare minne bor i $q,\mu$ (§5.4), inte i individer.

**A3 — Första ordningen.** Rörelse är härledd ur avstånd till mål; ingen egen fart, ingen eftersväng. Andra ordningen införs bara mot evidens om cykler.

**A4 — Sådd ∝ förändring, uppgifter beständiga.** Sådden följer förträngningens *takt*; det sådda är beständigt; död är en avläsning (platsvärde → 0), inte en ventil.

**A5 — Två avtappningsvägar, exogena, ev. oanvända.** Mättnad (efterfrågetak + omsättningstakt via $\eta$) och nästa fält (via $a(\mathbf r)$). Båda finns redan; ingen ny mekanism. Efterfrågan som upphör av *annan* teknisk utveckling är en exogen efterfrågeförskjutning, utanför modellens räckvidd.

**A6 — Efterfrågetak per riktning.** Exogent, av ett enda slag (återkommande aptit med omsättningstakt; "garderob" och "mat" är samma sort vid olika takt). Populationen ser aldrig taket direkt — bara att mättade riktningar slutar löna sig. $\eta$ blir avståndet till väggen, inte en fri svept parameter.

**A7 — Framväxande teknologier exogena.** Modellören sätter när, var och hur fält tänds och mognar. Ekonomin reagerar men föder inte fält. Flera överlappande vågor tillåts; deras interaktion är ett fynd. Endogen innovation är ett separat experiment.

**A8 — Total arbetskraft fast.** Omfördelning mellan yrken finns fullt ut; in-/utträde och demografi är parkerat.

**A9 — Augmenteringens karaktärsglidning endogen** via prisgrinden + $A_K(t)$ (§6). Den frigjorda tidens öde är experimentratten $\alpha$.

**A10 — Nyfött yrke = $(\mu, R_o)$ ur klumpen.** Centroid och task radius läses av den obundna massans tyngdpunkt och spridning med Paper 1:s egna ekvationer (samma operator, klump i stället för uppgifter). Förmågeprofilen infereras över regionen av radie $R_o$ (simuleringslager, flaggat).

*Varifrån arbetarna kommer.* Det nyfödda yrket föds med massa — den obundna klump som korsade tröskeln, via bindningsflödet $U \to b_o$. Det är inte tomt och väntar på att befolkas. Under **medflyttande origo** (A1) finns inget fast $L^0$ att saknas ur; origo är nuvarande $L(t)$, och yrket går in i fördelningen vid födseln med sin bindningsmassa, varefter omsorteringen kan föra mer dit. (Under fryst origo vore detta skarpare: yrket saknas i baslinjen och kan bara nås via $P(\text{nytt}\mid o')$ — ännu ett skäl att medflyttande origo är det koherenta valet.)

*Den dyraste platsen mot koden.* Yrkesfödelse bryter den fasta yrkesmängd som worker-lagrets prekomputer bakar in: `self.e` (readiness), `self.d` (avstånd), `L0` är alla $n_{\text{occ}}$-dimensionella och fasta. När $n_{\text{occ}}$ växer måste de byggas om mitt i förloppet. Detta är den enda platsen där dynamiken inte är en omparametrisering av befintligt arbete utan en strukturell ombyggnad, och där "minsta tillägg"-disciplinen är mest spänd.

**A11 — Två skalor, åtskilda.** $R_o$ = mobilitetsmetrik (Paper 2: median normaliserad övergång $\approx 1.03\,R_o$). $\ell$ = kapacitetsgrind, **hålls oförändrad** ($\ell\approx 0.133$). Uppmätt: grindens halvräckvidd $\approx 2\,R_o$ (median $2.2$, spridning $1.7$–$2.9$ över yrken). Faktor två är ett approximativt mått på skillnaden mellan *kunna* (kapacitetshölje) och *göra* (realiserad rörlighet) — en brygga mellan Paper 2 och Paper 3, **inte** en parameter som pressas ihop och **inte** en mekanism-ratt. *Talet $2.2$ mättes redan i skivenheter på båda sidor (grindens räckvidd lästes rumsligt); men dess **tolkning** som en geometrisk storhet — att $\ell$, som lever i deficit-enheter, översätts till en rumslig räckvidd — förutsätter Ö2. Bryggan hänger alltså på Ö2.*

**A12 — Riktad grind behålls.** Deficiten är ensidig ($\max(\cdot,0)$) och $v$-viktad (anisotrop), till skillnad från det symmetriska, isotropa centroidavståndet. Denna riktning genererar Paper 2:s uppmätta tvärsystem-asymmetri (35 mot 20). Kalibreras illustrativt; ingen falsk precision ärvs från pandemiperioden 2020–2024.

*Tillägg (dynamisk bindning).* Den ensidiga grinden behålls för den statiska kapaciteten och för tvärsystem-asymmetrin. Bindnings-readiness i den dynamiska modellen (§5.3, $e_o^{\text{match}}$) är däremot *tvåsidig* — den straffar även överkvalifikation — eftersom *kunna* (statiskt hölje) och *bli hem för* (dynamisk bindning) är olika frågor. De två grindarna samexisterar med olika roller; ingen ersätter den andra.

### Öppna frågor som rör grunden

**Ö1 — Prisfältet $\Pi$.** I dag fryst (mätlager). Över decennier rör sig priser. Om $\Pi$ ska bli ett rörligt fält är det ett stort, separat, grundberörande beslut. **Ej avgjort.**

**Ö2 — Reduktion deficit → avstånd (bärande för A11).** Sats att bevisa: under rang-2 är $q_k$ affint i positionen, så deficit-grinden är ett (ensidigt, viktat) euklidiskt avstånd. Detta är **inte** valfritt: A11:s faktor två lever på en översättning mellan deficit-enheter ($v$-viktade förmågeenheter) och skivavstånd, och det är Ö2 som rättfärdigar att de får mätas i jämförbara enheter. Utan Ö2 är "$2R_o$" ett deskriptivt förhållande över heterogena enheter, inte en geometrisk sats — och bryggan mellan Paper 2 och Paper 3 är då svagare än A11 låter påskina. Bör bevisas, inte skjutas upp.

**Ö3 — Survival-grinden: absolut kostnad mot komparativ fördel.** Överlevnadsvillkoret $(1-a)$ låter reinstateringen överleva där $a$ är låg, dvs. där människan är billigare än maskinen vid *uniform* kostnad $R$ — vilket är låg-$\Pi$-terräng (billiga uppgifter). Den överlevande sådden hamnar därför i lågvärdesarbete, och utfallet lutar mot lågkompetens (förstärkt av den trängda regimen, §9.7). Men AR-reinstatering är nya *komplexa* uppgifter där arbetskraften har komparativ fördel — ofta *högre* kompetens. För att fånga det måste överlevnaden styras av *relativ* produktivitet: ett varierande mänskligt fält $h(\mathbf r)$, högt där maskinen är svag (exponeringens komplement, eller de icke-automatiserbara förmågorna), i stället för absolut kostnad mot uniform $R$. Stort, separat, grundberörande beslut. **Ej avgjort.** *(Definitionen av $h$ är själva den ekonomiska handlingen och får inte sättas för att ge AR-resultatet; den ska grundas i förmågestrukturen.)*

---

## 8. Tidsskalor

Omsortering $\theta_L$ (år), mognad $1/\rho_i$ (år–decennium), karaktärsglidning $\theta_q$ (decennier). $U$ fylls snabbt och töms i den långsammaste takten av dem som kan binda. **Gapen mellan skalorna *är* dynamiken** — det är för att lagren rör sig olika fort som gap öppnar sig.

---

## 9. Nya observabler som statiken inte kan visa

1. **Pucklen i obundenhet $U_{\text{tot}}(t)$** — gapet mellan förstörelse och skapande, som transient. Huvudfyndet.
2. **Förträngning-leder-reinstatement-fördröjning** — kapaciteten $C$ läses av nuvarande $L$, så bindning väntar på att folk flyttar.
3. **Arbetsandelens bana $\Lambda(t)$** med fasdekomposition — vilken kanal dominerar i varje fas, och inflektionernas ordning.
4. **Bessens uppgång-och-fall som förlopp** — via $\eta$ som avstånd till taket (A6), härlett av mognad + mättnad snarare än handritat.
5. **Lead-lag mellan överlappande vågor** — ett nytt fält förtränger det ett tidigare fält skapade; nyfödda yrkens bredd $R_o$ styr deras exponering för nästa våg.
6. **Gapet mellan de två jämvikterna** — historiens inlåsning, mätt som avståndet mellan den statiska jämvikten (löst som fixpunkt, fräsch-allokering) och den dynamiska vilopunkten (nollpunkt för rörelselagen, nådd genom integration från startbilden). Två olika beräkningar, inte två punkter i samma.

7. **Tempokvoten avgör destinationen, inte bara transienten.** Pucklen (§9.1) mäter övergångens *kostnad*; men samma kvot — chocktempo $T_{\text{shock}}$ mot omfördelningstempo $\theta$ — avgör också *vem som till slut gör det reinstaterade arbetet*. Under en **gradvis** chock ($\theta < T_{\text{shock}}$) hinner bästa match med, och arbetet binds till de högkompetensspecialister vars förmåga passar fronten. Under en **trängd** chock ($\theta \gg T_{\text{shock}}$) hinner de små bästa-match-yrken inte binda (litet tak $c_o = M_o/\theta$); massan hopar sig i $U$ och sugs i stället upp av de stora yrken som *kan* absorbera snabbt — en tvingad lågkompetensnedgradering. Slutfördelningarna i de två regimerna korrelerar endast $\approx 0.30$: samma teknologi, samma fält, olika *samhälleligt* utfall beroende på takten. Där §9.6 mäter gapet mot fräsch-allokeringen, visar detta att den dynamiska vilopunkten *själv* förskjuts med tempot. Det är destinations-motstycket till pucklen, och en slutsats jämviktsramverket inte kan ge. *(Rör inte grunden; faller ut av bindningslagen §5.3 plus kalendertempot §5.2.)*

---

## 10. Nästa steg

**Gjort sedan förra revisionen (21 juni 2026).** $U$ är byggt som lager (§5.3) med en match-allokerad, storleks-taktbegränsad bindning; $U_{\text{tot}}(t)$ är instrumenterad över ett kalenderförankrat $A_K(t)$-förlopp (femårschock); pucklen (§9.1) och tempoberoende-destinationen (§9.7) är reproducerade. Förra revisionens falsifiering — "rör sig $U$ knappt, är effekten kosmetisk" — föll ut åt andra hållet: $U$ rör sig, och dess *destination* skiftar med tempot.

Kvarstår:

- **Mät gapet (§9.6)** från samma teknologi: fräsch-allokeringens jämvikt mot den krypande dynamikens vilopunkt.
- **Kalibrera $\theta_{\text{abs}}$ och $\beta$** och svep tempokvoten $T_{\text{shock}}/\theta$ systematiskt (§9.7); avgör om absorptionskapacitet skalar linjärt eller sublinjärt med storlek.
- **Avgör Ö3** (survival-grinden: absolut kostnad mot komparativ fördel) — om reinstateringen ska kunna landa i högkompetensarbete krävs ett varierande $h(\mathbf r)$ i stället för uniform $R$.
- **Hårdna $q$-glidningen (§5.4) och augmenteringsratten $\alpha$ (§6).**
- **Bär empirisk/hypotetisk-flaggan** (§2) genom hela pipelinen.
- **Avgör Ö1 ($\Pi$)** och **bevisa Ö2 (deficit → avstånd)** vid behov.