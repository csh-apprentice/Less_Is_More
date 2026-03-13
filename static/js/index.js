/* =========================================================
   Less Is More — Project Page JS
   Tab switching + video synchronization
   ========================================================= */

(function () {
  "use strict";

  /* -------------------------------------------------------
     1. Gallery tab switching
     ------------------------------------------------------- */
  function initTabs() {
    var tabBtns = document.querySelectorAll(".gallery-tab-btn");
    var panels = document.querySelectorAll(".gallery-panel");

    tabBtns.forEach(function (btn) {
      btn.addEventListener("click", function () {
        var target = btn.getAttribute("data-tab");

        tabBtns.forEach(function (b) {
          b.classList.remove("is-active");
        });
        panels.forEach(function (p) {
          p.classList.remove("is-active");
        });

        btn.classList.add("is-active");
        var panel = document.getElementById("tab-" + target);
        if (panel) {
          panel.classList.add("is-active");
        }
      });
    });
  }

  /* -------------------------------------------------------
     2. Video synchronisation
     Strategy:
       - Group videos by data-sync-group attribute.
       - Use IntersectionObserver to detect when a group's
         container section enters the viewport.
       - Once all videos in a group have loadeddata, start
         them simultaneously via play() calls after setting
         currentTime = 0.
       - Every 5 s, check for drift: if any video in the
         group has drifted more than 0.15 s from the
         leader (index 0), snap it back.
     ------------------------------------------------------- */

  var DRIFT_THRESHOLD = 0.15;   // seconds
  var DRIFT_CHECK_INTERVAL = 5000; // ms
  var syncGroups = {};          // groupId -> [videoElement, ...]

  function collectSyncGroups() {
    var videos = document.querySelectorAll("video[data-sync-group]");
    videos.forEach(function (v) {
      var gid = v.getAttribute("data-sync-group");
      if (!syncGroups[gid]) {
        syncGroups[gid] = [];
      }
      syncGroups[gid].push(v);
    });
  }

  function allReady(videos) {
    return videos.every(function (v) {
      return v.readyState >= 2; // HAVE_CURRENT_DATA
    });
  }

  function startGroup(videos) {
    videos.forEach(function (v) {
      v.currentTime = 0;
    });
    videos.forEach(function (v) {
      var p = v.play();
      if (p !== undefined) {
        p.catch(function () {
          // Autoplay blocked — do nothing; muted videos should autoplay fine.
        });
      }
    });
  }

  function correctDrift(videos) {
    if (videos.length < 2) return;
    var leader = videos[0];
    if (leader.paused) return;
    var leaderTime = leader.currentTime;
    for (var i = 1; i < videos.length; i++) {
      var v = videos[i];
      if (Math.abs(v.currentTime - leaderTime) > DRIFT_THRESHOLD) {
        v.currentTime = leaderTime;
      }
    }
  }

  function waitAndStart(videos) {
    if (allReady(videos)) {
      startGroup(videos);
      return;
    }
    var pending = videos.filter(function (v) {
      return v.readyState < 2;
    });
    var loaded = 0;
    pending.forEach(function (v) {
      function onReady() {
        loaded++;
        if (loaded === pending.length) {
          startGroup(videos);
        }
        v.removeEventListener("loadeddata", onReady);
        v.removeEventListener("canplay", onReady);
      }
      v.addEventListener("loadeddata", onReady);
      v.addEventListener("canplay", onReady);
    });
  }

  function initVideoSync() {
    collectSyncGroups();

    Object.keys(syncGroups).forEach(function (gid) {
      var videos = syncGroups[gid];

      // Find an ancestor container to observe
      var container = videos[0].closest("[data-sync-section]") ||
                      videos[0].closest("section") ||
                      videos[0].parentElement;

      var started = false;

      var observer = new IntersectionObserver(
        function (entries) {
          entries.forEach(function (entry) {
            if (entry.isIntersecting && !started) {
              started = true;
              waitAndStart(videos);
              observer.disconnect();
            }
          });
        },
        { threshold: 0.1 }
      );

      observer.observe(container);
    });

    // Periodic drift correction
    setInterval(function () {
      Object.keys(syncGroups).forEach(function (gid) {
        correctDrift(syncGroups[gid]);
      });
    }, DRIFT_CHECK_INTERVAL);
  }

  /* -------------------------------------------------------
     3. Init on DOMContentLoaded
     ------------------------------------------------------- */
  document.addEventListener("DOMContentLoaded", function () {
    initTabs();
    initVideoSync();
  });
})();
