// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "About",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-cv",
          title: "CV",
          description: "My academic CV.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "nav-publications",
          title: "Publications",
          description: "My publications in reversed chronological order.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-portfolio",
          title: "Portfolio",
          description: "My research projects and work.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/portfolio/";
          },
        },{id: "nav-blog",
          title: "Blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-photography",
          title: "Photography",
          description: "A collection of my photography work.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/photography/";
          },
        },{id: "post-how-to-think-about-scalability-in-system",
        
          title: "How to Think about scalability in system",
        
        description: "Understanding scalability through concrete examples, complexity analysis, and practical techniques. Build intuition instead of memorizing concepts.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/scalability-in-system/";
          
        },
      },{id: "post-from-model-to-system-how-memory-rag-tools-agents-and-mcp-fit-together",
        
          title: "From Model to System: How Memory, RAG, Tools, Agents, and MCP Fit Together...",
        
        description: "A system-level explanation of how LLM components—memory, RAG, tools, agents, skills, MCP, and the production harness—connect into one coherent architecture.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/llm-system-architecture-from-model-to-production/";
          
        },
      },{id: "post-the-market-is-not-reacting-to-reality",
        
          title: "The Market Is Not Reacting to Reality",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/market-expectations-vs-reality/";
          
        },
      },{id: "post-the-state-of-ai-a-deep-structural-analysis-2024-2026",
        
          title: "The State of AI: A Deep Structural Analysis (2024–2026)",
        
        description: "Macro view of the AI stack, agents, data, deployment, and industry dynamics — long-form analysis.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/ai-industry-analysis-2024-2026/";
          
        },
      },{id: "post-sem-layout-gan-mle-technical-notes",
        
          title: "SEM ↔ Layout GAN — MLE Technical Notes",
        
        description: "Structured narrative, mock Q&amp;A, and deep-dive follow-ups for SEM/layout GAN systems.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/sem-layout-gan-mle-notes/";
          
        },
      },{id: "post-transformers-amp-vision-transformers-a-deep-technical-guide",
        
          title: "Transformers &amp; Vision Transformers: A Deep Technical Guide",
        
        description: "Deep-dive notes on attention, ViT, engineering trade-offs, and common pitfalls.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/transformers-vit-deep-dive/";
          
        },
      },{id: "post-gans-for-image-to-image-translation-an-engineering-perspective",
        
          title: "GANs for Image-to-Image Translation: An Engineering Perspective",
        
        description: "What it actually takes to build a production GAN system — from architecture choices to training stability to deployment feedback loops.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/gans-image-to-image-translation-engineering/";
          
        },
      },{id: "post-vectors-vs-pixels-two-ways-to-search-for-the-same-thing-in-geometric-data",
        
          title: "Vectors vs. Pixels: Two Ways to Search for the Same Thing in Geometric...",
        
        description: "How the same spatial search problem can be solved with coordinate-based indexing or image-based morphology, and what the trade-offs teach us about algorithm design.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/vectors-vs-pixels-geometric-search/";
          
        },
      },{id: "post-semiconductor-data-is-not-like-other-data-a-practical-guide-for-ml-and-data-engineers",
        
          title: "Semiconductor Data Is Not Like Other Data: A Practical Guide for ML and...",
        
        description: "What makes manufacturing metrology data unique, how it differs from the datasets most ML practitioners are used to, and what to do about it.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/semiconductor-data-practical-guide/";
          
        },
      },{id: "post-beyond-rendering-rasterization-topology-and-the-bridge-to-search",
        
          title: "Beyond Rendering: Rasterization, Topology, and the Bridge to Search",
        
        description: "Why rasterization in engineering systems is a data transformation—not a drawing exercise—and how topology preservation enables reliable downstream search.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/rasterization-topology-bridge-to-search/";
          
        },
      },{id: "post-running-a-modern-python-cv-stack-on-a-15-year-old-linux-server",
        
          title: "Running a Modern Python CV Stack on a 15-Year-Old Linux Server",
        
        description: "Lessons from deploying NumPy, SciPy, and OpenCV in an offline legacy engineering environment",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/legacy-linux-python-deployment/";
          
        },
      },{id: "news-published-my-first-blog-post-running-a-modern-python-cv-stack-on-a-15-year-old-linux-server-lessons-from-deploying-numpy-scipy-and-opencv-in-an-offline-legacy-engineering-environment",
          title: 'Published my first blog post: Running a Modern Python CV Stack on a...',
          description: "",
          section: "News",},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%78%69%61%6F%73%69%7A%68%61%6E%67%30%35%31%37@%67%6D%61%69%6C.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/xiaosi-zhang-97948a201", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=N0-TpGcAAAAJ", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/xiaosi0517", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
