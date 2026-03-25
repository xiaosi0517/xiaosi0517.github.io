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
          section: "News",},{id: "projects-graphene-retinal-electrodes",
          title: 'Graphene Retinal Electrodes',
          description: "Transparent graphene microelectrode arrays for probing light-stimulated retinal activities.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project/";
            },},{id: "projects-graphene-laser-processing",
          title: 'Graphene Laser Processing',
          description: "In situ monitoring of graphene properties during laser-induced morphological changes.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/2_project/";
            },},{
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
