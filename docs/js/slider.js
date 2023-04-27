var limit_slider = document.getElementById("limit");
var peak_reduction_slider = document.getElementById("peak_reduction");

var output_limit = document.getElementById("limit_val");
output_limit.innerHTML = limit_slider.value;

var output_peak = document.getElementById("peak_reduction_val");
output_peak.innerHTML = peak_reduction_slider.value;

var source_combobox = document.getElementById("source");

var output_audio = document.getElementById("compressor");
var output_source = document.getElementById("compressor_source");
var input_audio = document.getElementById("input");
var input_source = document.getElementById("input_source");
var bypass_button = document.getElementById("bypass");
var play_button = document.getElementById("play");


var src = "fvox";
var limit = 1;
var peak = 80;
var model = "tcn-300-c";
var active = "output";

limit_slider.oninput = function() {
    limit = this.value;
    output_limit.innerHTML = limit;
    var filepath = `audio/demo/${src}-${limit}-${peak}-${model}.mp3`;
    output_source.src = filepath;
    if (input_audio.paused) {
        var time = output_audio.currentTime;
        output_audio.load();
        output_audio.currentTime = time;
        output_audio.play();
        active = "output";
    }
    else {
        var time = input_audio.currentTime;
        input_audio.load();
        input_audio.currentTime = time;
        input_audio.play();
        active = "input";
    }
}

peak_reduction_slider.oninput = function() {
    peak = this.value;
    output_peak.innerHTML = peak;
    var filepath = `audio/demo/${src}-${limit}-${peak}-${model}.mp3`;
    output_source.src = filepath;
    if (input_audio.paused) {
        var time = output_audio.currentTime;
        output_audio.load();
        output_audio.currentTime = time;
        output_audio.play();
        active = "output";
    }
    else {
        var time = input_audio.currentTime;
        input_audio.load();
        input_audio.currentTime = time;
        input_audio.play();
        active = "input";
    }
}

source_combobox.oninput = function() {
    input_audio.pause();
    output_audio.pause();
    src = this.value;
    var filepath = `audio/demo/${src}-${limit}-${peak}-${model}.mp3`;
    var inputfilepath = `audio/demo/${src}.mp3`;
    output_source.src = filepath;
    input_source.src = inputfilepath;
    output_audio.load();
    input_audio.load();
}

bypass_button.onclick = function() {
    if (active == "output") {
        bypass_button.innerHTML = "Bypassed"
        var playPromise = output_audio.pause();
        input_audio = document.getElementById("input");
        output_audio = document.getElementById("compressor");
        input_audio.currentTime = output_audio.currentTime;
        console.log(input_audio.currentTime);
        if (!(input_audio.paused && output_audio.paused)) {
            input_audio.play();
            active = "input";
        }
    }
    else {
        bypass_button.innerHTML = "On"
        var playPromise = input_audio.pause();
        input_audio = document.getElementById("input");
        output_audio = document.getElementById("compressor");
        output_audio.currentTime = input_audio.currentTime;
        console.log(output_audio.currentTime);
        if (!(input_audio.paused && output_audio.paused)) {
            output_audio.play();
            active = "output";
        }
    }
}

play_button.onclick = function() {
    if (input_audio.paused && output_audio.paused) {
        play_button.innerHTML = "Playing";
        if (bypass_button.innerHTML = "On") {
            output_audio.play();
            active = "output";
        }
        else {
            output_audio.pause();
            input_audio.play();
            active = "input";
        }
    }
    else {
        play_button.innerHTML = "Stopped";
        output_audio.pause();
        input_audio.pause();
    }
}