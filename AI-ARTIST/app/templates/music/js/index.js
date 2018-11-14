(function() {
  var AudioAnalyser, Visualizer, VisualizerBar, vis;

  Visualizer = class Visualizer {
    constructor(size) {
      this.windowResize = this.windowResize.bind(this);
      this.render = this.render.bind(this);
      this.size = size;
      this.levels = [];
      this.buildScene();
      this.buildGrid();
	  //this.stream = document.createElement("AUDIO");
	  //this.stream.setAttribute("src", "For River.mp3");
      window.addEventListener('resize', this.windowResize);
      this.render();
	  $('.start').on('click', (e) => {
		 var ref;
        if ((ref = this.analyser) != null) {
          ref.stop();
        }
		this.stream = document.createElement("AUDIO");
		this.stream.setAttribute("src", "For River.mp3");
        //this.stream = URL.createObjectURL($('input#fileselect')[0].files[0]);
        return this.startAnalyser();

      });

      $('input#fileselect').on('change', (e) => {
        var ref;
        if ((ref = this.analyser) != null) {
          ref.stop();
        }
		this.stream = document.createElement("AUDIO");
		this.stream.setAttribute("src", "For River.mp3");
        //this.stream = URL.createObjectURL($('input#fileselect')[0].files[0]);
        return this.startAnalyser();
      });
    }

    play(data) {
      this.stream = data;
      return this.startAnalyser();
    }

    buildScene() {
      this.scene = new THREE.Scene();
      this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
      this.renderer = new THREE.WebGLRenderer({
        antialias: true
      });
      this.renderer.setSize(window.innerWidth, window.innerHeight);
      document.body.appendChild(this.renderer.domElement);
      this.scene.add(new THREE.AmbientLight(0x303030));
      this.light = new THREE.DirectionalLight(0xffffff, 1);
      this.scene.add(this.light);
      this.positionLight();
      return this.positionCamera();
    }

    windowResize() {
      this.renderer.setSize(window.innerWidth, window.innerHeight);
      this.camera.aspect = window.innerWidth / window.innerHeight;
      return this.camera.updateProjectionMatrix();
    }

    buildGrid() {
      var bar, col, row;
      this.grid = new THREE.Object3D();
      this.scene.add(this.grid);
      return this.bars = (function() {
        var i, ref, results;
        results = [];
        for (row = i = 0, ref = this.size; (0 <= ref ? i < ref : i > ref); row = 0 <= ref ? ++i : --i) {
          results.push((function() {
            var j, ref1, results1;
            results1 = [];
            for (col = j = 0, ref1 = this.size; (0 <= ref1 ? j < ref1 : j > ref1); col = 0 <= ref1 ? ++j : --j) {
              bar = new VisualizerBar(row, col, this.size);
              this.grid.add(bar.mesh);
              results1.push(bar);
            }
            return results1;
          }).call(this));
        }
        return results;
      }).call(this);
    }

    startAnalyser() {
      this.analyser = new AudioAnalyser(this.stream, this.size, 0.5);
      this.analyser.onUpdate = (bands) => {
        return this.updateLevels(bands);
      };
      this.analyser.start();
      $('.pause').on('click', () => {
        return this.analyser.stop();
      });
      return $('.play').on('click', () => {
        return this.analyser.start();
      });
    }

    positionCamera() {
      this.camera.position.y = 18;
      this.camera.position.x = 20;
      this.camera.position.z = 0;
      this.camera.lookAt(this.scene.position);
      return this.camera.position.y = 10;
    }

    positionLight() {
      this.light.position.y = 2;
      this.light.position.x = 0;
      return this.light.position.z = -1;
    }

    updateLevels(bands) {
      this.levels.unshift(Array.prototype.slice.call(bands));
      if (this.levels.length > 16) {
        return this.levels.pop();
      }
    }

    updateBars() {
      var bar, i, len, ref, results, row, x, y;
      ref = this.bars;
      results = [];
      for (x = i = 0, len = ref.length; i < len; x = ++i) {
        row = ref[x];
        results.push((function() {
          var j, len1, ref1, results1;
          results1 = [];
          for (y = j = 0, len1 = row.length; j < len1; y = ++j) {
            bar = row[y];
            results1.push(bar.setLevel((ref1 = this.levels[x]) != null ? ref1[y] : void 0));
          }
          return results1;
        }).call(this));
      }
      return results;
    }

    render(t = 0) {
      requestAnimationFrame(this.render);
      this.updateBars();
      this.grid.rotation.y = t / 3000;
      return this.renderer.render(this.scene, this.camera);
    }

  };

  VisualizerBar = class VisualizerBar {
    constructor(row, col, size) {
      var geometry, material;
      this.level = 0;
      this.row = row;
      this.col = col;
      this.size = size;
      this.spacing = 1.8;
      this.scale_factor = 3;
      this.offset = this.size * this.spacing / 2 - 1;
      material = new THREE.MeshLambertMaterial({
        color: this.color(),
        ambient: this.color()
      });
      geometry = new THREE.BoxGeometry(1, 1, 1);
      this.mesh = new THREE.Mesh(geometry, material);
      this.mesh.position.set(this.xPos(), 0, this.zPos());
      this.setLevel();
    }

    xPos() {
      return this.row * this.spacing - this.offset;
    }

    zPos() {
      return this.col * this.spacing - this.offset;
    }

    setLevel(l = 0.1) {
      if (l < 0.1) {
        l = 0.1;
      }
      if (this.level !== l / 255) {
        this.level = l / 255;
        this.mesh.scale.y = this.level * this.scale_factor;
        return this.mesh.position.y = this.level * this.scale_factor / 2;
      }
    }

    color() {
      var b, g, r, s;
      s = 255 / (this.size + 1) * 1.3;
      g = 255 - Math.ceil(this.col * s);
      b = 255 - Math.ceil(this.row * s);
      r = (200 - Math.ceil((this.row + this.col) / 2 * s * 1.5)) * -1;
      b = b < 0 ? 0 : b;
      g = g < 0 ? 0 : g;
      r = r < 0 ? 0 : r;
      return r * 65536 + g * 256 + b;
    }

  };

  AudioAnalyser = (function() {
    class AudioAnalyser {
      constructor(audio = new Audio(), numBands = 256, smoothing = 0.3) {
        var src;
        this.audio = audio;
        this.numBands = numBands;
        this.smoothing = smoothing;
        
        // construct audio object
        if (typeof this.audio === 'string') {
          src = this.audio;
          this.audio = new Audio();
          this.audio.controls = true;
          this.audio.src = src;
        }
        
        // setup audio context and nodes
        this.context = new AudioAnalyser.AudioContext();
        
        // createScriptProcessor so we can hook onto updates
        this.jsNode = this.context.createScriptProcessor(1024, 1, 1);
        
        // smoothed analyser with n bins for frequency-domain analysis
        this.analyser = this.context.createAnalyser();
        this.analyser.smoothingTimeConstant = this.smoothing;
        this.analyser.fftSize = this.numBands * 2;
        
        // persistant bands array
        this.bands = new Uint8Array(this.analyser.frequencyBinCount);
        // circumvent http://crbug.com/112368
        this.audio.addEventListener('canplay', () => {
          
          // media source
          this.source = this.context.createMediaElementSource(this.audio);
          // wire up nodes
          this.source.connect(this.analyser);
          this.analyser.connect(this.jsNode);
          this.jsNode.connect(this.context.destination);
          this.source.connect(this.context.destination);
          // update each time the JavaScriptNode is called
          return this.jsNode.onaudioprocess = () => {
            // retreive the data from the first channel
            this.analyser.getByteFrequencyData(this.bands);
            if (!this.audio.paused) {
              return typeof this.onUpdate === "function" ? this.onUpdate(this.bands) : void 0;
            }
          };
        });
      }

      start() {
        $('.controls').addClass('playing');
        return this.audio.play();
      }

      stop() {
        $('.controls').removeClass('playing');
        return this.audio.pause();
      }

    };

    //# Stole this class from soulwire
    //# https://codepen.io/soulwire/pen/Dscga
    AudioAnalyser.AudioContext = self.AudioContext || self.webkitAudioContext;

    AudioAnalyser.enabled = AudioAnalyser.AudioContext != null;

    return AudioAnalyser;

  }).call(this);

  vis = new Visualizer(16);

  window.loadData = (data) => {
    $('.overlay .title').text('Infected Mushroom - Symphonatic');
    return vis.play(data);
  };

}).call(this);

//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiPGFub255bW91cz4iXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBRUE7QUFBQSxNQUFBLGFBQUEsRUFBQSxVQUFBLEVBQUEsYUFBQSxFQUFBOztFQUFNLGFBQU4sTUFBQSxXQUFBO0lBQ0UsV0FBYSxDQUFDLElBQUQsQ0FBQTtVQW9DYixDQUFBLG1CQUFBLENBQUE7VUFpREEsQ0FBQSxhQUFBLENBQUE7TUFwRkUsSUFBQyxDQUFBLElBQUQsR0FBUTtNQUNSLElBQUMsQ0FBQSxNQUFELEdBQVU7TUFFVixJQUFDLENBQUEsVUFBRCxDQUFBO01BQ0EsSUFBQyxDQUFBLFNBQUQsQ0FBQTtNQUVBLE1BQU0sQ0FBQyxnQkFBUCxDQUF3QixRQUF4QixFQUFrQyxJQUFDLENBQUEsWUFBbkM7TUFFQSxJQUFDLENBQUEsTUFBRCxDQUFBO01BRUEsQ0FBQSxDQUFFLGtCQUFGLENBQXFCLENBQUMsRUFBdEIsQ0FBeUIsUUFBekIsRUFBbUMsQ0FBQyxDQUFELENBQUEsR0FBQTtBQUNqQyxZQUFBOzthQUFTLENBQUUsSUFBWCxDQUFBOztRQUNBLElBQUMsQ0FBQSxNQUFELEdBQVUsR0FBRyxDQUFDLGVBQUosQ0FBb0IsQ0FBQSxDQUFFLGtCQUFGLENBQXNCLENBQUEsQ0FBQSxDQUFFLENBQUMsS0FBTSxDQUFBLENBQUEsQ0FBbkQ7ZUFDVixJQUFDLENBQUEsYUFBRCxDQUFBO01BSGlDLENBQW5DO0lBWFc7O0lBZ0JiLElBQU0sQ0FBQyxJQUFELENBQUE7TUFDSixJQUFDLENBQUEsTUFBRCxHQUFVO2FBQ1YsSUFBQyxDQUFBLGFBQUQsQ0FBQTtJQUZJOztJQUlOLFVBQVksQ0FBQSxDQUFBO01BQ1YsSUFBQyxDQUFBLEtBQUQsR0FBUyxJQUFJLEtBQUssQ0FBQyxLQUFWLENBQUE7TUFDVCxJQUFDLENBQUEsTUFBRCxHQUFVLElBQUksS0FBSyxDQUFDLGlCQUFWLENBQTZCLEVBQTdCLEVBQWlDLE1BQU0sQ0FBQyxVQUFQLEdBQW9C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O1dBQ0EsR0FBRyxDQUFDLElBQUosQ0FBUyxJQUFUO0VBRmdCO0FBN01sQiIsInNvdXJjZXNDb250ZW50IjpbIlxuXG5jbGFzcyBWaXN1YWxpemVyXG4gIGNvbnN0cnVjdG9yOiAoc2l6ZSkgLT4gIFxuICAgIEBzaXplID0gc2l6ZVxuICAgIEBsZXZlbHMgPSBbXVxuICAgICBcbiAgICBAYnVpbGRTY2VuZSgpXG4gICAgQGJ1aWxkR3JpZCgpXG4gICAgXG4gICAgd2luZG93LmFkZEV2ZW50TGlzdGVuZXIgJ3Jlc2l6ZScsIEB3aW5kb3dSZXNpemVcblxuICAgIEByZW5kZXIoKVxuICAgIFxuICAgICQoJ2lucHV0I2ZpbGVzZWxlY3QnKS5vbiAnY2hhbmdlJywgKGUpID0+XG4gICAgICBAYW5hbHlzZXI/LnN0b3AoKVxuICAgICAgQHN0cmVhbSA9IFVSTC5jcmVhdGVPYmplY3RVUkwoJCgnaW5wdXQjZmlsZXNlbGVjdCcpWzBdLmZpbGVzWzBdKVxuICAgICAgQHN0YXJ0QW5hbHlzZXIoKVxuICAgICAgXG4gIHBsYXk6IChkYXRhKSAtPiBcbiAgICBAc3RyZWFtID0gZGF0YVxuICAgIEBzdGFydEFuYWx5c2VyKClcbiAgICBcbiAgYnVpbGRTY2VuZTogLT5cbiAgICBAc2NlbmUgPSBuZXcgVEhSRUUuU2NlbmUoKVxuICAgIEBjYW1lcmEgPSBuZXcgVEhSRUUuUGVyc3BlY3RpdmVDYW1lcmEoIDc1LCB3aW5kb3cuaW5uZXJXaWR0aCAvIHdpbmRvdy5pbm5lckhlaWdodCwgMC4xLCAxMDAwIClcbiAgICBcbiAgICBAcmVuZGVyZXIgPSBuZXcgVEhSRUUuV2ViR0xSZW5kZXJlclxuICAgICAgYW50aWFsaWFzOiB0cnVlXG4gICAgQHJlbmRlcmVyLnNldFNpemUoIHdpbmRvdy5pbm5lcldpZHRoLCB3aW5kb3cuaW5uZXJIZWlnaHQgKVxuICAgIFxuICAgIGRvY3VtZW50LmJvZHkuYXBwZW5kQ2hpbGQoIEByZW5kZXJlci5kb21FbGVtZW50IClcbiAgICBAc2NlbmUuYWRkIG5ldyBUSFJFRS5BbWJpZW50TGlnaHQoIDB4MzAzMDMwIClcbiAgICBcbiAgICBAbGlnaHQgPSBuZXcgVEhSRUUuRGlyZWN0aW9uYWxMaWdodCggMHhmZmZmZmYsIDEgKTtcbiAgICBAc2NlbmUuYWRkKCBAbGlnaHQgKVxuICAgIEBwb3NpdGlvbkxpZ2h0KClcbiAgICBAcG9zaXRpb25DYW1lcmEoKVxuICAgIFxuICB3aW5kb3dSZXNpemU6ID0+XG4gICAgQHJlbmRlcmVyLnNldFNpemUoIHdpbmRvdy5pbm5lcldpZHRoLCB3aW5kb3cuaW5uZXJIZWlnaHQgKVxuICAgIFxuICAgIEBjYW1lcmEuYXNwZWN0ID0gd2luZG93LmlubmVyV2lkdGggLyB3aW5kb3cuaW5uZXJIZWlnaHRcbiAgICBAY2FtZXJhLnVwZGF0ZVByb2plY3Rpb25NYXRyaXgoKVxuICAgIFxuICBidWlsZEdyaWQ6IC0+XG4gICAgQGdyaWQgPSBuZXcgVEhSRUUuT2JqZWN0M0QoKVxuICAgIEBzY2VuZS5hZGQgQGdyaWRcbiAgICBAYmFycyA9IGZvciByb3cgaW4gWyAwLi4uQHNpemUgXVxuICAgICAgZm9yIGNvbCBpbiBbIDAuLi5Ac2l6ZSBdXG4gICAgICAgIGJhciA9IG5ldyBWaXN1YWxpemVyQmFyKHJvdyxjb2wsQHNpemUpXG4gICAgICAgIEBncmlkLmFkZChiYXIubWVzaClcbiAgICAgICAgYmFyXG4gICAgICAgIFxuICBzdGFydEFuYWx5c2VyOiAtPlxuICAgIEBhbmFseXNlciA9IG5ldyBBdWRpb0FuYWx5c2VyKEBzdHJlYW0sQHNpemUsMC41KSBcbiAgICBAYW5hbHlzZXIub25VcGRhdGUgPSAoYmFuZHMpID0+XG4gICAgICBAdXBkYXRlTGV2ZWxzIGJhbmRzICAgIFxuICAgIEBhbmFseXNlci5zdGFydCgpICAgICAgICBcbiAgICAkKCcucGF1c2UnKS5vbiAnY2xpY2snLCA9PlxuICAgICAgQGFuYWx5c2VyLnN0b3AoKVxuXG4gICAgJCgnLnBsYXknKS5vbiAnY2xpY2snLCA9PlxuICAgICAgQGFuYWx5c2VyLnN0YXJ0KClcbiAgICAgIFxuICBwb3NpdGlvbkNhbWVyYTogLT5cbiAgICBAY2FtZXJhLnBvc2l0aW9uLnkgPSAxOFxuICAgIEBjYW1lcmEucG9zaXRpb24ueCA9IDIwXG4gICAgQGNhbWVyYS5wb3NpdGlvbi56ID0gMFxuICAgIEBjYW1lcmEubG9va0F0IEBzY2VuZS5wb3NpdGlvbiBcbiAgICBAY2FtZXJhLnBvc2l0aW9uLnkgPSAxMFxuICAgIFxuICBwb3NpdGlvbkxpZ2h0OiAtPlxuICAgIEBsaWdodC5wb3NpdGlvbi55ID0gMlxuICAgIEBsaWdodC5wb3NpdGlvbi54ID0gMFxuICAgIEBsaWdodC5wb3NpdGlvbi56ID0gLTFcbiAgICBcbiAgICBcbiAgdXBkYXRlTGV2ZWxzOiAoYmFuZHMpIC0+XG4gICAgQGxldmVscy51bnNoaWZ0IEFycmF5LnByb3RvdHlwZS5zbGljZS5jYWxsKGJhbmRzKVxuICAgIGlmIEBsZXZlbHMubGVuZ3RoID4gMTZcbiAgICAgIEBsZXZlbHMucG9wKCkgXG5cbiAgdXBkYXRlQmFyczogLT5cbiAgICBmb3Igcm93LHggaW4gQGJhcnNcbiAgICAgIGZvciBiYXIseSBpbiByb3dcbiAgICAgICAgYmFyLnNldExldmVsKEBsZXZlbHNbeF0/W3ldKVxuICAgICAgICAgIFxuICByZW5kZXI6ICh0PTApID0+XG4gICAgcmVxdWVzdEFuaW1hdGlvbkZyYW1lIEByZW5kZXIgXG4gICAgQHVwZGF0ZUJhcnMoKVxuICAgIEBncmlkLnJvdGF0aW9uLnkgPSB0LzMwMDBcbiAgICBAcmVuZGVyZXIucmVuZGVyIEBzY2VuZSwgQGNhbWVyYSBcblxuICAgIFxuICAgICBcbmNsYXNzIFZpc3VhbGl6ZXJCYXJcbiAgY29uc3RydWN0b3I6IChyb3csY29sLHNpemUpIC0+XG4gICAgQGxldmVsID0gMFxuICAgIEByb3cgPSByb3dcbiAgICBAY29sID0gY29sXG4gICAgQHNpemUgPSBzaXplXG4gICAgQHNwYWNpbmcgPSAxLjhcbiAgICBAc2NhbGVfZmFjdG9yID0gM1xuICAgIEBvZmZzZXQgPSBAc2l6ZSAqIEBzcGFjaW5nIC8gMiAtIDEgXG4gICAgbWF0ZXJpYWwgPSBuZXcgVEhSRUUuTWVzaExhbWJlcnRNYXRlcmlhbFxuICAgICAgY29sb3I6IEBjb2xvcigpXG4gICAgICBhbWJpZW50OiBAY29sb3IoKVxuICAgICAgXG4gICAgZ2VvbWV0cnkgPSBuZXcgVEhSRUUuQm94R2VvbWV0cnkoIDEsMSwxIClcbiAgICBAbWVzaCA9IG5ldyBUSFJFRS5NZXNoKCBnZW9tZXRyeSwgbWF0ZXJpYWwgKVxuICAgIEBtZXNoLnBvc2l0aW9uLnNldCggQHhQb3MoKSwgMCwgQHpQb3MoKSApXG4gICAgQHNldExldmVsKClcbiAgICBcbiAgeFBvczogLT5cbiAgICBAcm93KkBzcGFjaW5nLUBvZmZzZXRcbiAgICBcbiAgelBvczogLT5cbiAgICBAY29sKkBzcGFjaW5nLUBvZmZzZXRcbiAgICBcbiAgc2V0TGV2ZWw6IChsPTAuMSkgLT5cbiAgICBpZiBsIDwgMC4xIFxuICAgICAgbCA9IDAuMVxuICAgIGlmIEBsZXZlbCBpc250IGwvMjU1IFxuICAgICAgQGxldmVsID0gbC8yNTVcbiAgICAgIFxuICAgICAgQG1lc2guc2NhbGUueSA9IEBsZXZlbCpAc2NhbGVfZmFjdG9yXG4gICAgICBAbWVzaC5wb3NpdGlvbi55ID0gQGxldmVsKkBzY2FsZV9mYWN0b3IvMlxuICAgIFxuICBjb2xvcjogLT5cbiAgICBzPTI1NS8oQHNpemUrMSkqMS4zXG4gICAgXG4gICAgZyA9IDI1NS1NYXRoLmNlaWwoQGNvbCpzKVxuICAgIGIgPSAyNTUtTWF0aC5jZWlsKEByb3cqcylcbiAgICByID0gKDIwMC1NYXRoLmNlaWwoKEByb3crQGNvbCkvMipzKjEuNSkpKi0xXG4gICAgXG5cbiAgICBiID0gaWYgYiA8IDAgdGhlbiAwIGVsc2UgYlxuICAgIGcgPSBpZiBnIDwgMCB0aGVuIDAgZWxzZSBnXG4gICAgciA9IGlmIHIgPCAwIHRoZW4gMCBlbHNlIHIgICBcbiAgICBcbiAgICByKjY1NTM2ICsgZyoyNTYgKyBiXG4gICAgXG5jbGFzcyBBdWRpb0FuYWx5c2VyXG4gICMjIFN0b2xlIHRoaXMgY2xhc3MgZnJvbSBzb3Vsd2lyZVxuICAjIyBodHRwczovL2NvZGVwZW4uaW8vc291bHdpcmUvcGVuL0RzY2dhXG4gIFxuICBAQXVkaW9Db250ZXh0OiBzZWxmLkF1ZGlvQ29udGV4dCBvciBzZWxmLndlYmtpdEF1ZGlvQ29udGV4dFxuICBAZW5hYmxlZDogQEF1ZGlvQ29udGV4dD9cbiAgXG4gIGNvbnN0cnVjdG9yOiAoIEBhdWRpbyA9IG5ldyBBdWRpbygpLCBAbnVtQmFuZHMgPSAyNTYsIEBzbW9vdGhpbmcgPSAwLjMgKSAtPlxuICBcbiAgICAjIGNvbnN0cnVjdCBhdWRpbyBvYmplY3RcbiAgICBpZiB0eXBlb2YgQGF1ZGlvIGlzICdzdHJpbmcnXG4gICAgICBcbiAgICAgIHNyYyA9IEBhdWRpb1xuICAgICAgQGF1ZGlvID0gbmV3IEF1ZGlvKClcbiAgICAgIEBhdWRpby5jb250cm9scyA9IHllc1xuICAgICAgQGF1ZGlvLnNyYyA9IHNyY1xuICBcbiAgICAjIHNldHVwIGF1ZGlvIGNvbnRleHQgYW5kIG5vZGVzXG4gICAgQGNvbnRleHQgPSBuZXcgQXVkaW9BbmFseXNlci5BdWRpb0NvbnRleHQoKVxuICAgIFxuICAgICMgY3JlYXRlU2NyaXB0UHJvY2Vzc29yIHNvIHdlIGNhbiBob29rIG9udG8gdXBkYXRlc1xuICAgIEBqc05vZGUgPSBAY29udGV4dC5jcmVhdGVTY3JpcHRQcm9jZXNzb3IgMTAyNCwgMSwgMVxuICAgIFxuICAgICMgc21vb3RoZWQgYW5hbHlzZXIgd2l0aCBuIGJpbnMgZm9yIGZyZXF1ZW5jeS1kb21haW4gYW5hbHlzaXNcbiAgICBAYW5hbHlzZXIgPSBAY29udGV4dC5jcmVhdGVBbmFseXNlcigpXG4gICAgQGFuYWx5c2VyLnNtb290aGluZ1RpbWVDb25zdGFudCA9IEBzbW9vdGhpbmdcbiAgICBAYW5hbHlzZXIuZmZ0U2l6ZSA9IEBudW1CYW5kcyAqIDJcbiAgICBcbiAgICAjIHBlcnNpc3RhbnQgYmFuZHMgYXJyYXlcbiAgICBAYmFuZHMgPSBuZXcgVWludDhBcnJheSBAYW5hbHlzZXIuZnJlcXVlbmN5QmluQ291bnRcblxuICAgICMgY2lyY3VtdmVudCBodHRwOi8vY3JidWcuY29tLzExMjM2OFxuICAgIEBhdWRpby5hZGRFdmVudExpc3RlbmVyICdjYW5wbGF5JywgPT5cbiAgICBcbiAgICAgICMgbWVkaWEgc291cmNlXG4gICAgICBAc291cmNlID0gQGNvbnRleHQuY3JlYXRlTWVkaWFFbGVtZW50U291cmNlIEBhdWRpb1xuXG4gICAgICAjIHdpcmUgdXAgbm9kZXNcblxuICAgICAgQHNvdXJjZS5jb25uZWN0IEBhbmFseXNlclxuICAgICAgQGFuYWx5c2VyLmNvbm5lY3QgQGpzTm9kZVxuXG4gICAgICBAanNOb2RlLmNvbm5lY3QgQGNvbnRleHQuZGVzdGluYXRpb25cbiAgICAgIEBzb3VyY2UuY29ubmVjdCBAY29udGV4dC5kZXN0aW5hdGlvblxuXG4gICAgICAjIHVwZGF0ZSBlYWNoIHRpbWUgdGhlIEphdmFTY3JpcHROb2RlIGlzIGNhbGxlZFxuICAgICAgQGpzTm9kZS5vbmF1ZGlvcHJvY2VzcyA9ID0+XG5cbiAgICAgICAgIyByZXRyZWl2ZSB0aGUgZGF0YSBmcm9tIHRoZSBmaXJzdCBjaGFubmVsXG4gICAgICAgIEBhbmFseXNlci5nZXRCeXRlRnJlcXVlbmN5RGF0YSBAYmFuZHNcbiAgICAgICAgXG4gICAgICAgICMgZmlyZSBjYWxsYmFja1xuICAgICAgICBAb25VcGRhdGU/IEBiYW5kcyBpZiBub3QgQGF1ZGlvLnBhdXNlZFxuICAgICAgICBcbiAgc3RhcnQ6IC0+XG4gICAgJCgnLmNvbnRyb2xzJykuYWRkQ2xhc3MgJ3BsYXlpbmcnXG4gICAgQGF1ZGlvLnBsYXkoKVxuICAgIFxuICBzdG9wOiAtPlxuICAgICQoJy5jb250cm9scycpLnJlbW92ZUNsYXNzICdwbGF5aW5nJ1xuICAgIEBhdWRpby5wYXVzZSgpXG5cbnZpcyA9IG5ldyBWaXN1YWxpemVyKDE2KSAgXG5cbndpbmRvdy5sb2FkRGF0YSA9IChkYXRhKSA9PlxuICAkKCcub3ZlcmxheSAudGl0bGUnKS50ZXh0ICdJbmZlY3RlZCBNdXNocm9vbSAtIFN5bXBob25hdGljJ1xuICB2aXMucGxheSBkYXRhIl19
//# sourceURL=coffeescript
