<h1>Modulation Extraction for LFO-driven Audio Effects</h1>
<p>
    <a href="https://christhetr.ee/" target=”_blank”>Christopher Mitcheltree</a>,
    <a href="https://www.christiansteinmetz.com/" target=”_blank”>Christian J. Steinmetz</a>,
    <a href="https://mcomunita.github.io/" target=”_blank”>Marco Comunità</a>, and
    <a href="https://www.eecs.qmul.ac.uk/~josh/" target=”_blank”>Joshua D. Reiss</a>
</p>

<hr>
<h2>Links</h2>

<h3><a href="https://arxiv.com" target=”_blank”>Paper</a></h3>

<h3><a href="https://christhetree.github.io/mod_extraction/" target=”_blank”>Supplemental Figures, Listening Samples, and Plugins</a></h3>

<hr>
<h2>Repository Instructions</h2>

<ol>
    <li>Clone this repository and open its directory.</li>
    <li>Create an out directory (<code>mkdir out</code>).</li>
    <li>Create a data directory (<code>mkdir data</code>).</li>
    <li>
    Install the requirements using <br><code>conda env create --file=conda_env_cpu.yml</code> or <br>
    <code>conda env create --file=conda_env.yml</code><br> for GPU acceleration. <br>
    <code>requirements_pipchill.txt</code> and <code>requirements_all.txt</code> are also provided as references, 
    but are not needed when using <code>conda</code>.
    </li>
    <li>Data instructions TBD</li>
    <li>The source code can be explored in the <code>mod_extraction/</code> directory.</li>
    <li>All models from the paper can be found in the <code>models/</code> directory.</li>
    <li>
    All models can be trained by modifying <code>scripts/train.py</code> and the corresponding 
    <code>train_ ... .yml</code> config file and then running <code>python scripts/train.py</code>. <br>
    Make sure your PYTHONPATH has been set correctly by running a command like 
    <code>export PYTHONPATH=$PYTHONPATH:BASE_DIR/mod_extraction/</code>.
    </li>
    <li>
    All models can be evaluated by modifying <code>scripts/validate.py</code> and the corresponding 
    <code>eval_ ... .yml</code> config file and then running <code>python scripts/validate.py</code>.
    </li>
    <li>
    <a href="https://neutone.space" target=”_blank”>Neutone</a> files for running the effect models as a VST can be  
    exported by modifying and running the <code>scripts/export_neutone_models.py</code> file.
    </li>
</ol>

<hr>
<h2>Abstract</h2>

<p>
Low frequency oscillator (LFO) driven audio effects such as phaser, flanger, and chorus, modify
their input using time-varying filters and delays, resulting in characteristic sweeping or widening
effects.
It has been shown that these effects can be modeled using neural networks that are conditioned with
the ground truth LFO signal. However, in most cases this signal is not accessible and cannot be
easily measured from the output audio.
To address this, we propose a neural network that can accurately extract arbitrary LFO signals from
processed audio for multiple digital audio effects, parameter settings, and instrument
configurations.
Since our system imposes no restrictions on the LFO signal shape, we demonstrate its ability to
extract quasiperiodic, combined, and distorted modulation signals that are relevant to analog effect
modeling.
Furthermore, we show how coupling the extraction model with a simple processing network enables
training of end-to-end black-box models of unseen analog or digital LFO-driven audio effects from
just dry and wet audio pairs - hence overcoming the need to access the audio effect plugin or
internal LFO signal.
We make our code available and provide the trained audio effect models in a real-time VST plugin.
</p>

<hr>
<h2>Citation</h2>

<div>
<pre><code>
@misc{mitcheltree2023modulation,
      title={Modulation Extraction for LFO-driven Audio Effects},
      author={Christopher Mitcheltree and Christian J. Steinmetz and Marco Comunità and Joshua D. Reiss},
      year={2023},
      eprint={TBD},
      archivePrefix={arXiv},
      primaryClass={cs.SD}}
</code></pre>
</div>
