## ☕ Buy me a coffee

Se ti piace il progetto e vuoi supportarlo, puoi offrirmi un caffè tramite [PayPal](https://paypal.me/nicovic17) oppure [Revolut](https://revolut.me/nicoauriemma).

# Webcam Memes

All'avvio di `run3.py` scegli una delle tre modalita, con il mouse o con i
tasti `1`, `2`, `3`:

| Selezione | Funzionamento |
|---|---|
| **Meme Webcam** | Il Meme Pose Tracker gia presente: stessi gesti, meme, riquadri e indicatori. |
| **Gojo Webcam** | Unlimited Void: ambiente cosmico animato attorno alla persona, con audio. |
| **Sukuna Webcam** | Malevolent Shrine: santuario dietro alla persona, teschi animati davanti al busto e audio. |

I due domini sono copie autonome degli effetti sviluppati in `disappear/local_app`.
Moduli, modelli e asset necessari sono dentro **webcam_memes**: per avviarli
non occorre accedere all'altro progetto. Il menu compare prima dell'apertura
della webcam. `Q` o `ESC` nel menu annullano l'avvio.

`run3.py` è un prototipo Python che riconosce alcune pose delle mani tramite webcam e sovrappone automaticamente un meme al volto dell’utente. Il risultato viene mostrato in una finestra locale e pubblicato contemporaneamente come **webcam virtuale OBS**, così può essere usato in applicazioni come Microsoft Teams, Zoom, Discord o Google Meet.

L’elaborazione utilizza MediaPipe per rilevare volto e mani, OpenCV per acquisire e comporre i fotogrammi e `pyvirtualcam` per inviare il video alla webcam virtuale.

## Funzionalità

- rilevamento in tempo reale di un volto e fino a due mani;
- riconoscimento di otto pose tramite i landmark MediaPipe nella modalita Meme;
- sovrapposizione del meme associato direttamente sul volto;
- riquadri animati intorno alle mani rilevate;
- stabilizzazione temporale per limitare attivazioni accidentali e sfarfallii;
- indicatore della confidenza del rilevamento;
- anteprima locale a `960 × 540` e 30 FPS;
- pubblicazione dello stesso risultato su OBS Virtual Camera;
- elaborazione completamente locale, senza invio dei fotogrammi a server esterni.

## Pose riconosciute

| Posa | Gesto richiesto | Immagine mostrata |
|---|---|---|
| Due mani sul volto | Due mani aperte, verticali e ravvicinate davanti al volto | `memes/image.png` |
| Ascolto | Una mano aperta accanto a un orecchio | `memes/hearing.png` |
| Mani dietro la testa | Due mani separate ai lati della testa | `memes/memeomg.png` |
| Preghiera | Due mani giunte o intrecciate | `memes/sad.png` |
| Indicazione su di sé | Una mano sotto il volto con l’indice rivolto verso il petto | `memes/me.png` |
| Timeout | Due mani aperte e perpendicolari, disposte a forma di T | `memes/stop.png` |
| Calma | Una mano aperta davanti al petto, con quattro dita distese e abbastanza unite | `memes/calma.jpg` |
| Pollice alzato | Una mano chiusa con il pollice rivolto verso l'alto | `memes/memeok.jpg` |

Le pose hanno priorità diverse per ridurre le sovrapposizioni tra gesti simili. In genere una posa deve essere riconosciuta per almeno cinque fotogrammi consecutivi prima di attivarsi; la posa “Preghiera” richiede due fotogrammi. Il meme viene rimosso dopo otto fotogrammi senza una posa valida.

## Requisiti

- Windows 10 o Windows 11;
- Python 3 compatibile con MediaPipe;
- una webcam funzionante;
- OBS Studio con il componente **OBS Virtual Camera** installato;
- i modelli MediaPipe `face_landmarker.task` e `hand_landmarker.task`;
- le immagini presenti nella cartella `memes/`.
- per i domini, il modello `models/selfie_segmenter_landscape.tflite` e gli asset
  audio/immagine nella cartella `assets/`;
- `pygame` per gli audio; in alternativa il programma supporta `ffplay` nel PATH.
- `Pillow` per preparare le notifiche animate del dominio Gojo (incluso in `requirements.txt`).

## Installazione

Apri PowerShell nella cartella del progetto e installa le dipendenze:

```powershell
python -m pip install -r requirements.txt
```

Se `python` non viene riconosciuto, installa Python e abilita l’opzione per aggiungerlo al `PATH` durante l’installazione.

Installa inoltre OBS Studio. La webcam virtuale deve essere disponibile nel sistema, ma non deve essere già occupata da OBS o da un’altra istanza del programma quando avvii `run3.py`.

## Struttura dei file

La cartella deve contenere almeno questi file:

```text
webcam_memes/
├── run3.py
├── face_landmarker.task
├── hand_landmarker.task
└── memes/
    ├── image.png
    ├── hearing.png
    ├── memeomg.png
    ├── sad.png
    ├── me.png
    ├── stop.png
    └── calma.jpg
```

I percorsi vengono calcolati rispetto alla posizione di `run3.py`, quindi il programma può essere avviato anche da una directory di lavoro diversa.

Per i domini sono inclusi anche:

```text
webcam_domains.py
gojo_effect.py
gojo_comic.py
gojo_notifications.py
sukuna_effect.py
sukuna_skulls.py
vanish_sound.py
requirements.txt
models/selfie_segmenter_landscape.tflite
assets/gojo_domain.MP3
assets/unlimitedvoidcomic.png
assets/malevolent_shrine.png
assets/melevolent_shrine.MP3
```

Il modello delle mani `hand_landmarker.task` gia presente viene condiviso tra
le tre modalita. Gojo genera il cosmo proceduralmente; Sukuna usa il PNG
trasparente del santuario e disegna i teschi in codice. L'audio Sukuna ha il
nome fornito `melevolent_shrine.MP3`; e supportato anche `malevolent_shrine.mp3`.

## Avvio

Dalla cartella del progetto esegui:

```powershell
python run3.py
```

Il programma:

1. mostra il menu **Meme Webcam / Gojo Webcam / Sukuna Webcam**;
2. carica i modelli e gli asset della sola modalita scelta;
3. prova ad aprire la webcam con DirectShow, Media Foundation e infine con il backend automatico;
4. avvia OBS Virtual Camera a `960 × 540` e fino a 30 FPS;
5. mostra l'anteprima: **Meme Pose Tracker**, **Gojo Webcam** o **Sukuna Webcam**.

Premi `Q` oppure `Esc` per terminare il programma.

In tutte e tre le modalita trovi in basso al centro dell'anteprima il pulsante
**Specchio ON/OFF**. Cliccalo, oppure premi `M`, per invertire orizzontalmente
la webcam. Lo specchio e inizialmente attivo, come nelle versioni precedenti.
La scelta vale anche per OBS Virtual Camera; il pulsante compare soltanto
nell'anteprima locale. Vignette, scritte e asset degli effetti restano leggibili.

### Gojo Webcam

Mostra una mano alla webcam con **indice e medio alzati a V**, anulare e
mignolo ripiegati, anche in modo rilassato. Il pollice puo stare come ti e
piu comodo. Mantieni il segno per circa **0,4 secondi**: non serve incrociare
le dita, ma anche il gesto incrociato continua a funzionare. Il rilevamento
accetta entrambe le mani e una leggera piegatura delle dita. Brevi cali di
tracciamento fino a 0,35 secondi non azzerano subito il tempo di conferma.

Il dominio si apre gradualmente in circa 3,8 secondi: stelle, nebulose e un
buco nero animato sostituiscono la stanza, mentre la tua immagine continua
a muoversi. `gojo_domain.MP3` parte una volta all'attivazione.

Durante l'ingresso attraversi un tunnel di notifiche ispirate a Microsoft
Teams: schede viola con avatar, mittenti e messaggi diversi, che arrivano
dal centro e si ingrandiscono passando ai lati. Nel dominio aperto una nuova
notifica passa sullo sfondo circa ogni 2 secondi, con movimento e dissolvenza.
Le notifiche restano dietro alla persona e si richiudono insieme al dominio.
Sono messaggi inventati, disegnati localmente: non leggono notifiche reali
e non richiedono un account Teams. I testi si possono personalizzare in
`MESSAGES` dentro `gojo_notifications.py`. Le schede vengono preparate una
volta all'avvio della modalita, senza ridisegnare i testi a ogni fotogramma.

All'apertura compare anche `assets/unlimitedvoidcomic.png`, la vignetta
trasparente "Unlimited Void!": resta circa 3,2 secondi, con ingresso e uscita
graduali. Rimane nella meta destra dello schermo, accanto alla testa:
il rilevamento del volto ne aggiorna la posizione con un movimento smussato.
Appare sia con il gesto sia con `V`, anche sulla webcam
virtuale. Ogni nuova apertura la ripropone; `R` e `SPAZIO` la rimuovono.

### Sukuna Webcam

Avvicina le due mani davanti al petto con almeno due dita rivolte verso
l'alto per mano (per esempio indice e medio), oppure usa le mani a preghiera.
Possono restare leggermente separate: non serve unire le punte o sistemare
i pollici. Mantieni il segno per circa **0,45 secondi**. Nel terminale
compaiono automaticamente suggerimenti per aiutarti a posizionare le mani.

In circa 4,6 secondi compare il santuario **dietro** di te, con acqua scura,
riflessi, nebbia e cenere in movimento. In basso, **davanti** al busto,
emergono gradualmente 15 teschi cornuti in stile fumetto, con mandibole
animate. L'audio del dominio parte una volta all'attivazione.

Anche Sukuna include il tunnel di notifiche stile Teams durante l'ingresso:
le schede si avvicinano, si ingrandiscono e passano ai lati. A dominio aperto
i messaggi attraversano periodicamente lo sfondo con dissolvenze, dietro alla
persona e ai teschi in primo piano. Condivide grafica, testi e ottimizzazioni
di `gojo_notifications.py` con Gojo; sono notifiche inventate, senza accesso
a messaggi o account Teams reali. `R` e `SPAZIO` le rimuovono insieme al dominio.

### Comandi dei domini

| Tasto | Azione |
|---|---|
| `V` | Prova l'apertura senza il gesto. |
| `R` | Richiude il dominio; durante l'ingresso lo annulla. |
| `SPAZIO` | Azzera immediatamente effetto e riconoscimento del gesto. |
| `F` | Attiva/disattiva il fullscreen. |
| `M` | Attiva/disattiva lo specchio, come il pulsante nell'anteprima. |
| `Q` / `ESC` | Chiude il programma e libera webcam e audio. |

Rilascia il gesto prima di una nuova attivazione. Non serve acquisire uno
sfondo vuoto. `R`, `SPAZIO` e l'uscita interrompono l'audio; ripetere il gesto
o `V` a dominio gia attivo non sovrappone altre riproduzioni.
I domini mostrano un video pulito, con istruzioni nel terminale; riquadri e
indicatori originali rimangono nella sola modalita Meme Webcam.
Per scegliere un'altra modalita, chiudi e riavvia `run3.py`.

## Utilizzo nelle videochiamate

Dopo aver avviato `run3.py`, apri l’applicazione di videochiamata e seleziona **OBS Virtual Camera** come sorgente video.

Avvia prima `run3.py` e poi l’applicazione che deve ricevere il video. Non avviare contemporaneamente la Virtual Camera dall’interfaccia di OBS, perché un solo programma alla volta può controllare il dispositivo virtuale.

## Personalizzare le immagini

Puoi sostituire un meme mantenendo invariati nome, estensione e posizione del file. Per usare nomi differenti, modifica le costanti all’inizio di `run3.py`:

```python
MEME_PATH = BASE_DIR / "memes" / "image.png"
HEARING_MEME_PATH = BASE_DIR / "memes" / "hearing.png"
MEMEOMG_PATH = BASE_DIR / "memes" / "memeomg.png"
SAD_MEME_PATH = BASE_DIR / "memes" / "sad.png"
ME_MEME_PATH = BASE_DIR / "memes" / "me.png"
STOP_MEME_PATH = BASE_DIR / "memes" / "stop.png"
CALMA_MEME_PATH = BASE_DIR / "memes" / "calma.jpg"
```

Sono consigliate immagini nitide, con proporzioni adatte al volto. Il programma le ridimensiona mantenendo le proporzioni e le ritaglia quando necessario per non uscire dal fotogramma.

## Configurazione video

Le principali impostazioni sono definite all’inizio di `run3.py`:

```python
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
VIRTUAL_CAMERA_FPS = 30
```

- `CAPTURE_WIDTH` e `CAPTURE_HEIGHT` indicano la risoluzione richiesta alla webcam fisica;
- `CAMERA_WIDTH` e `CAMERA_HEIGHT` definiscono anteprima e webcam virtuale;
- `VIRTUAL_CAMERA_FPS` definisce la frequenza della webcam virtuale.

Ridurre risoluzione o FPS può migliorare le prestazioni sui computer meno potenti. Se modifichi la risoluzione della webcam virtuale, assicurati che sia supportata dalle applicazioni nelle quali vuoi utilizzarla.

### Fluidita e ritardo video

L'acquisizione gira su un thread dedicato e conserva solo il fotogramma piu
recente: se un effetto richiede piu tempo, i frame arretrati vengono scartati.
Anche l'analisi di mani, volto e segmentazione gira su un worker separato:
il video usa l'ultimo risultato disponibile e continua a scorrere mentre
MediaPipe elabora il fotogramma successivo. L'attesa per il limite FPS tiene
conto dell'intero ciclo, compresa l'attesa della webcam.
L'anteprima e OBS mantengono la risoluzione configurata (`960 x 540` di default).
L'analisi di volto e mani usa una copia piu leggera, larga al massimo 640 pixel.

La webcam viene configurata richiedendo MJPEG **dopo** risoluzione e FPS:
alcuni driver tornano a YUY2 se il formato viene impostato prima. All'avvio
il terminale mostra il formato realmente negoziato; durante l'uso compare
ogni tre secondi una riga `[VIDEO]` con FPS effettivi e tempo dell'analisi.

Lo sfondo cosmico di Gojo viene calcolato a una risoluzione interna ridotta;
la persona e la vignetta rimangono alla risoluzione video. I teschi di Sukuna
usano sprite e maschere precalcolati con compositing nativo OpenCV. Durante
un dominio gia aperto il riconoscimento delle mani viene eseguito meno spesso,
mentre i gesti di attivazione mantengono la frequenza precedente.

Queste ottimizzazioni riducono il carico e impediscono l'accumulo di ritardo;
gli FPS effettivi dipendono comunque da webcam, computer e programmi aperti.
Il valore 30 e il limite di uscita, non una garanzia che il sensore produca
30 fotogrammi nuovi al secondo. La verifica locale del 2 novembre 2026 sulla
webcam GENERAL WEBCAM ha misurato circa 5 FPS in YUY2 a 1280x720 e circa
16 FPS in MJPEG; il ciclo con gli effetti attivi ha mantenuto circa 16 FPS.
Per applicare un aggiornamento, chiudi e riavvia `run3.py`.

## Risoluzione dei problemi

### La webcam non si apre

- chiudi Teams, Zoom, browser e altri programmi che potrebbero utilizzare la webcam fisica;
- controlla in **Impostazioni di Windows > Privacy e sicurezza > Fotocamera** che le applicazioni desktop abbiano il permesso di accedervi;
- prova a scollegare e ricollegare la webcam;
- verifica che la webcam funzioni nell’app Fotocamera di Windows.

### OBS Virtual Camera non si avvia

- verifica che OBS Studio e il componente Virtual Camera siano installati;
- arresta la Virtual Camera dall’interfaccia di OBS, se è già attiva;
- chiudi eventuali altre istanze di `run3.py`;
- riavvia OBS o Windows se il dispositivo virtuale rimane occupato.

### Un modello o un’immagine non viene trovato

Controlla che `face_landmarker.task`, `hand_landmarker.task` e tutti i file nella cartella `memes/` abbiano esattamente i nomi indicati. Il programma interrompe l’avvio se una risorsa obbligatoria non può essere letta.

### Le pose vengono riconosciute male

- usa un ambiente ben illuminato;
- mantieni volto e mani interamente nell’inquadratura;
- evita che le mani si confondano con lo sfondo;
- posiziona la webcam frontalmente;
- esegui il gesto in modo netto e mantienilo per qualche istante;
- evita movimenti troppo rapidi o mani sovrapposte, salvo per i gesti che lo richiedono.

### Il video è lento o scatta

- chiudi le applicazioni che utilizzano molta CPU o GPU;
- riduci `CAPTURE_WIDTH`, `CAPTURE_HEIGHT`, `CAMERA_WIDTH` e `CAMERA_HEIGHT`;
- prova una webcam o una porta USB differente;
- aggiorna i driver della webcam.

## Privacy

I fotogrammi della webcam vengono elaborati localmente. `run3.py` non salva registrazioni e non invia immagini o landmark a servizi online. Il video elaborato viene reso disponibile localmente attraverso OBS Virtual Camera alle applicazioni scelte dall’utente.

Le applicazioni di videochiamata possono applicare proprie condizioni e politiche sulla privacy al flusso ricevuto dalla webcam virtuale.

## Limitazioni

- Il riconoscimento si basa su regole geometriche e può produrre falsi positivi o falsi negativi.
- Le soglie possono richiedere modifiche in base a webcam, distanza, illuminazione e caratteristiche dell’utente.
- Il programma rileva un solo volto e al massimo due mani.
- L’assenza temporanea del volto viene tollerata per circa 1,2 secondi.
- Il progetto è un prototipo sperimentale e non è destinato a impieghi medici, biometrici, di sorveglianza o di sicurezza.

## Supporto al progetto

Se il progetto ti è utile e vuoi sostenerne lo sviluppo, puoi offrire un caffè tramite [PayPal](https://paypal.me/nicovic17) oppure [Revolut](https://revolut.me/nicoauriemma).
