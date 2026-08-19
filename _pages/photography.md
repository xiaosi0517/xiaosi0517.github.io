---
layout: page
permalink: /photography/
title: Photography
description: A collection of my photography work.
nav: true
nav_order: 5
---

<div class="photo-gallery">
  {% for photo in site.data.photos %}
    <div class="photo-card">
      <div class="photo-frame">
        <img
          src="/assets/img/photography/{{ photo.image }}"
          alt="{{ photo.location }}"
          class="photo-image{% if photo.type == 'wallpaper' %} photo-wallpaper{% endif %}"
        >
      </div>

      <p class="photo-caption">
        {{ photo.location }} <span class="photo-date">· {{ photo.date }}</span><br>
        {{ photo.caption }}
        {% if photo.caption_en %}
          <br><span class="photo-caption-en">{{ photo.caption_en }}</span>
        {% endif %}
      </p>
    </div>
  {% endfor %}
</div>
