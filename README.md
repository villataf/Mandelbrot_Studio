# Mandelbrot Studio

Esploratore dell'insieme di Mandelbrot scritto in Rust, per Windows, Linux e macOS
(processori Intel/AMD a 64 bit e ARM64).

## Cosa fa

- **Calcolo su processore o scheda grafica** (tasto `G`). Sul processore usa tutti i core in
  doppia precisione; sulla scheda grafica (Vulkan, Metal o DirectX 12) sceglie da solo il
  metodo in base allo zoom: singola precisione (velocissima) per gli zoom bassi e
  **perturbazione** per quelli profondi (un'orbita di riferimento precisa calcolata dal
  processore, e ogni pixel seguito come piccola differenza da essa in singola precisione:
  da 3 a 10 volte più veloce della doppia precisione); restano disponibili la doppia
  emulata e la doppia nativa. Se
  ci sono più schede grafiche si può scegliere quale usare. Se la scheda non basta per lo
  zoom richiesto, il calcolo passa da solo al processore.
- **Zoom con il mouse** (rotellina, rettangolo col tasto destro, doppio clic) **o con la
  tastiera** (`+`/`−`, frecce, `Home`). Zoom massimo circa 6·10¹¹×, il limite dei numeri a
  64 bit.
- **Coordinate**: si possono scrivere a mano, copiare/incollare, salvare e ricaricare da
  file (`.json`, con tutte le cifre) e tenere tra i preferiti. Quando si cambia posizione
  (preferiti, coordinate scritte, keyframe, vista iniziale) la vista "vola": zoom indietro,
  spostamento e zoom avanti, a velocità fissa (si può disattivare).
  Il volo segue lo stesso schema di mandelbrot.site: percorso di van Wijk e Nuij (ρ = 1,42),
  0,8 s per unità di percorso, partenza immediata e atterraggio dolce; le immagini
  precedenti restano visibili sotto, in scala, e ogni nuova immagine compare in dissolvenza.
- **Zoom animato**: rotellina, tasti `+`/`−`, frecce, doppio clic e rettangolo producono
  uno zoom breve e fluido (circa 0,3 s) che tiene fermo il punto indicato; più scatti
  rapidi si sommano. Il trascinamento segue il mouse senza ritardo.
- **Pannelli laterali riducibili** con i pulsanti « e » o con `F2` / `F3`.
- **Impostazioni**: modalità notte (o chiaro, o come il sistema; anche con il pulsante
  🌙/☀ in alto a destra) e posizione all'apertura (l'ultima usata o quella iniziale).
- **Iterazioni**: cursore logaritmico, valore esatto, e un **valore consigliato** calcolato
  analizzando la zona inquadrata (si può applicare con un clic, con il tasto `I` o in
  automatico).
- **Colori**: 10 palette predefinite, editor dei colori, distribuzione (lineare, radice,
  logaritmica), ripetizioni e spostamento; le palette personalizzate si salvano e si
  possono esportare/importare.
- **Immagini**: JPEG con qualità 1–100, TIFF senza compressione o con LZW, Deflate o
  PackBits, PNG; qualsiasi risoluzione (fino a 30000 px per lato) con antialiasing.
- **Animazioni a keyframe**: ogni keyframe memorizza posizione, zoom, iterazioni e palette;
  tra un keyframe e l'altro lo zoom è fluido e i colori sfumano (la palette può cambiare
  durante l'animazione, con in più il ciclo dei colori). Anteprima nella finestra.
- **Esportazione video** in varie risoluzioni (da 640×360 a 8K, o personalizzata) e fps:
  - **H.264** o **H.265/HEVC** in MP4, codificati dalla **scheda grafica** (NVIDIA NVENC,
    AMD AMF, Intel Quick Sync, Apple VideoToolbox, VA-API su Linux) con livello di
    qualità costante (0–51) o bitrate, e velocità di codifica;
  - **RAW** RGB non compresso (con un file `.txt` che ne descrive il formato);
  - sequenza di immagini **TIFF**.

## Requisiti

- Per H.264/H.265 serve **ffmpeg** (con i codificatori hardware, come le build complete
  per Windows, Homebrew su macOS, o il pacchetto della distribuzione su Linux). Il
  programma lo cerca accanto a sé, nel `PATH` e nelle cartelle abituali; altrimenti si
  indica il percorso nel pannello «Esporta video». All'avvio prova ogni codificatore e
  mostra quali funzionano davvero sul computer. Se nessun codificatore della scheda
  grafica funziona, lo dice e propone quello del processore.
- RAW, TIFF e le immagini non hanno bisogno di ffmpeg.

## Compilare

Serve Rust (https://rustup.rs).

```
cargo build --release
```

Il programma si trova in `target/release/`. Su Linux servono anche le librerie di sviluppo
di X11/Wayland (`libxkbcommon-dev libwayland-dev libx11-dev libxcursor-dev libxrandr-dev
libxi-dev`).

Per gli altri sistemi e processori (Windows ARM64, Linux ARM64, macOS Intel e Apple
Silicon) il file `.github/workflows/build.yml` compila automaticamente tutte e sei le
versioni su GitHub.

## Controlli diagnostici

- `mandelbrot-studio --diagnostica` elenca le schede grafiche, confronta i loro risultati
  con quelli del processore e prova i codificatori video.
- `mandelbrot-studio --prova-esportazione <cartella>` salva immagini JPEG/TIFF e una breve
  animazione in H.264, H.265, RAW e TIFF per verificare tutta la catena.

## Comandi

| Mouse | |
|---|---|
| Rotellina | zoom nel punto indicato |
| Trascina (sinistro) | sposta |
| Trascina (destro) | zoom sul rettangolo |
| Doppio clic sinistro / destro | ingrandisci / riduci 2× |
| Clic con la rotellina | centra |

| Tastiera | |
|---|---|
| `+` `−` `PagSu` `PagGiù` | zoom (con Maiusc a passi piccoli) |
| Frecce | sposta |
| `Home` | vista iniziale |
| `[` `]` | meno / più iterazioni |
| `I` | iterazioni consigliate |
| `G` | GPU / CPU |
| `K` | aggiungi keyframe |
| `Spazio` | anteprima animazione |
| `C` | copia coordinate |
| `Ctrl+S` / `Ctrl+Maiusc+S` / `Ctrl+O` | salva immagine / salva coordinate / carica coordinate |
| `F1` | guida |
| `F2` / `F3` | riduci o apri il pannello sinistro / destro |

Impostazioni, preferiti e palette personalizzate vengono salvati nella cartella di
configurazione dell'utente (`mandelbrot-studio`).

## Licenza

GNU GPL v3 o successiva (vedi `LICENSE`).
