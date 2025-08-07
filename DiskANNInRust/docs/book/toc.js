// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded affix "><a href="introduction.html">Introduction</a></li><li class="chapter-item expanded affix "><li class="part-title">User Guide</li><li class="chapter-item expanded "><a href="user-guide/getting-started.html"><strong aria-hidden="true">1.</strong> Getting Started</a></li><li class="chapter-item expanded "><a href="user-guide/installation.html"><strong aria-hidden="true">2.</strong> Installation</a></li><li class="chapter-item expanded "><a href="user-guide/cli.html"><strong aria-hidden="true">3.</strong> Command Line Interface</a></li><li class="chapter-item expanded "><a href="user-guide/api.html"><strong aria-hidden="true">4.</strong> Programming Interface</a></li><li class="chapter-item expanded affix "><li class="part-title">Architecture</li><li class="chapter-item expanded "><a href="architecture/overview.html"><strong aria-hidden="true">5.</strong> Overview</a></li><li class="chapter-item expanded "><a href="architecture/components.html"><strong aria-hidden="true">6.</strong> Core Components</a></li><li class="chapter-item expanded "><a href="architecture/vamana.html"><strong aria-hidden="true">7.</strong> Vamana Algorithm</a></li><li class="chapter-item expanded "><a href="architecture/beam-search.html"><strong aria-hidden="true">8.</strong> Beam Search</a></li><li class="chapter-item expanded "><a href="architecture/simd.html"><strong aria-hidden="true">9.</strong> SIMD Optimizations</a></li><li class="chapter-item expanded affix "><li class="part-title">Features</li><li class="chapter-item expanded "><a href="features/flags.html"><strong aria-hidden="true">10.</strong> Feature Flags</a></li><li class="chapter-item expanded "><a href="features/performance.html"><strong aria-hidden="true">11.</strong> Performance Tuning</a></li><li class="chapter-item expanded "><a href="features/memory.html"><strong aria-hidden="true">12.</strong> Memory Management</a></li><li class="chapter-item expanded "><a href="features/safety.html"><strong aria-hidden="true">13.</strong> Safety Guarantees</a></li><li class="chapter-item expanded affix "><li class="part-title">Examples</li><li class="chapter-item expanded "><a href="examples/build-index.html"><strong aria-hidden="true">14.</strong> Building an Index</a></li><li class="chapter-item expanded "><a href="examples/search.html"><strong aria-hidden="true">15.</strong> Searching Vectors</a></li><li class="chapter-item expanded "><a href="examples/batch.html"><strong aria-hidden="true">16.</strong> Batch Processing</a></li><li class="chapter-item expanded "><a href="examples/ffi.html"><strong aria-hidden="true">17.</strong> FFI Integration</a></li><li class="chapter-item expanded affix "><li class="part-title">Development</li><li class="chapter-item expanded "><a href="development/contributing.html"><strong aria-hidden="true">18.</strong> Contributing</a></li><li class="chapter-item expanded "><a href="development/testing.html"><strong aria-hidden="true">19.</strong> Testing</a></li><li class="chapter-item expanded "><a href="development/benchmarking.html"><strong aria-hidden="true">20.</strong> Benchmarking</a></li><li class="chapter-item expanded "><a href="development/profiling.html"><strong aria-hidden="true">21.</strong> Profiling</a></li><li class="chapter-item expanded affix "><li class="part-title">Reference</li><li class="chapter-item expanded "><a href="reference/api.html"><strong aria-hidden="true">22.</strong> API Reference</a></li><li class="chapter-item expanded "><a href="reference/formats.html"><strong aria-hidden="true">23.</strong> File Formats</a></li><li class="chapter-item expanded "><a href="reference/errors.html"><strong aria-hidden="true">24.</strong> Error Handling</a></li><li class="chapter-item expanded "><a href="reference/performance.html"><strong aria-hidden="true">25.</strong> Performance Characteristics</a></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split("#")[0].split("?")[0];
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
