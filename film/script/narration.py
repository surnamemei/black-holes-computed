"""Narration script.  Each segment: (id, text spoken by the TTS, subtitle text).

Spoken text spells numbers out and can override pronunciations with misaki's
inline syntax [word](/phonemes/).  Subtitles use normal typography.

Fact notes (checked while writing):
* M87 distance 16.8 Mpc = 54.8 Mly; mass 6.5e9 Msun (EHT 2019 Paper VI).
* Photon sphere r = 3M = 1.5 r_s; shadow radius sqrt(27) M -> diameter 2.6x horizon (Schwarzschild).
* ISCO orbital speed: 0.5c (a=0) rising to ~0.6c for a=0.9 (ZAMO-measured).
* r_s(6.5e9 Msun) = 128 AU; Neptune's orbit diameter ~60 AU.
* Ring diameter 42 +/- 3 uas (EHT 2019 Paper I); 42 uas at 384,400 km = 7.8 cm (an orange).
* lambda/D: 1.3 mm / 12,742 km = 1.0e-10 rad = 21 uas.
* EHT 2017: 8 stations at 6 sites (Hawaii: JCMT+SMA; Arizona: SMT; Mexico: LMT; Chile: ALMA+APEX;
  Spain: IRAM 30m; South Pole: SPT).  SPT cannot see M87 (dec +12 deg), so we do not claim it did.
* Hydrogen-maser timestamps; data shipped on hard drives to correlators (MIT Haystack, MPIfR Bonn).
* Four independent imaging teams worked blind (EHT 2019 Paper IV).
"""

SEGMENTS = [
    ("open1",
     "Fifty-five million light-years away, at the heart of the galaxy [Messier](/mˈɛsiˌA/) eighty-seven, lies a black hole six and a half billion times the mass of our Sun.",
     "55 million light-years away, at the heart of the galaxy Messier 87, lies a black hole 6.5 billion times the mass of our Sun."),
    ("open2",
     "In twenty nineteen, astronomers revealed the first picture of a black hole. This is how that picture was made, and why a black hole looks the way it does.",
     "In 2019, astronomers revealed the first picture of a black hole. This is how that picture was made, and why a black hole looks the way it does."),

    ("lens1",
     "Everything you will see here was computed from Einstein's theory of gravity, by tracing billions of rays of light through curved spacetime.",
     "Everything you will see here was computed from Einstein's theory of gravity, by tracing billions of rays of light through curved spacetime."),
    ("lens2",
     "Begin with empty space, and a field of stars. Now, add a black hole. Its gravity bends the light passing by. Stars behind it are stretched into arcs, duplicated, and folded into rings.",
     "Begin with empty space, and a field of stars. Now, add a black hole. Its gravity bends the light passing by. Stars behind it are stretched into arcs, duplicated, and folded into rings."),

    ("rays1",
     "To see why, follow individual rays of light. Rays passing far from the black hole bend only slightly. Closer in, they swing around sharply.",
     "To see why, follow individual rays of light. Rays passing far from the black hole bend only slightly. Closer in, they swing around sharply."),
    ("rays2",
     "At one and a half times the radius of the event horizon, light can travel in a circle around the black hole. This is the photon sphere. Rays that come any closer spiral in, and are lost.",
     "At 1.5 times the radius of the event horizon, light can travel in a circle around the black hole. This is the photon sphere. Rays that come any closer spiral in, and are lost."),
    ("rays3",
     "So a black hole casts a shadow. And because gravity bends light around its edge, the shadow looks two and a half times wider than the event horizon itself.",
     "So a black hole casts a shadow. And because gravity bends light around its edge, the shadow looks 2.5 times wider than the event horizon itself."),

    ("disk1",
     "Real black holes are rarely dark. Gas spiraling inward forms a disk, heated by friction until it glows.",
     "Real black holes are rarely dark. Gas spiraling inward forms a disk, heated by friction until it glows."),
    ("disk2",
     "Look closely, and something strange appears. We can see the far side of the disk, which should be hidden behind the black hole. Its light is bent up over the top, and down underneath.",
     "Look closely, and something strange appears. We can see the far side of the disk, which should be hidden behind the black hole. Its light is bent up over the top, and down underneath."),
    ("disk3",
     "And hugging the shadow is a thin, bright ring: light that looped around the black hole, once or more, before escaping toward us.",
     "And hugging the shadow is a thin, bright ring: light that looped around the black hole, once or more, before escaping toward us."),

    ("dopp1",
     "But this picture is still too symmetric. Near the black hole, the gas orbits at more than half the speed of light.",
     "But this picture is still too symmetric. Near the black hole, the gas orbits at more than half the speed of light."),
    ("dopp2",
     "On the side rushing toward us, its light is compressed: bluer, and far brighter. On the receding side, it is stretched: redder, and dimmer. Climbing out of the black hole's gravity robs every photon of energy, too.",
     "On the side rushing toward us, its light is compressed: bluer, and far brighter. On the receding side, it is stretched: redder, and dimmer. Climbing out of the black hole's gravity robs every photon of energy, too."),
    ("dopp3",
     "The real view is lopsided: one side blazing, the other fading into the dark.",
     "The real view is lopsided: one side blazing, the other fading into the dark."),

    ("spin1",
     "Black holes can also spin. A spinning black hole drags space itself around with it.",
     "Black holes can also spin. A spinning black hole drags space itself around with it."),
    ("spin2",
     "Light traveling with the spin can orbit closer in, while light traveling against it is pushed away. The shadow is flattened on one side.",
     "Light traveling with the spin can orbit closer in, while light traveling against it is pushed away. The shadow is flattened on one side."),

    ("scale1",
     "[Messier](/mˈɛsiˌA/) eighty-seven's black hole is a giant. Its event horizon is wider than the orbits of all the planets in our solar system.",
     "Messier 87's black hole is a giant. Its event horizon is wider than the orbits of all the planets in our solar system."),
    ("scale2",
     "But it is so far away that its shadow spans just forty-two [microarcseconds](/mˌIkɹOˈɑɹksˌɛkəndz/) on the sky. That's the size of an orange on the surface of the Moon, seen from Earth.",
     "But it is so far away that its shadow spans just 42 microarcseconds on the sky. That's the size of an orange on the surface of the Moon, seen from Earth."),
    ("scale3",
     "The finest detail a telescope can see is set by the wavelength it observes, divided by the size of its dish. At a wavelength of one point three millimeters, where the hot gas near the black hole turns transparent, resolving that shadow would take a dish as wide as the Earth.",
     "The finest detail a telescope can see is set by the wavelength it observes, divided by the size of its dish. At a wavelength of 1.3 millimeters, where the hot gas near the black hole turns transparent, resolving that shadow would take a dish as wide as the Earth."),

    ("eht1",
     "We can't build a telescope that big. But we can build one that is mostly missing.",
     "We can't build a telescope that big. But we can build one that is mostly missing."),
    ("eht2",
     "In April twenty seventeen, radio observatories in Hawaii, Arizona, Mexico, Chile, Spain, and at the South Pole, joined to form the Event Horizon Telescope. Each recorded radio waves from the sky, stamped with the time from an atomic clock.",
     "In April 2017, radio observatories in Hawaii, Arizona, Mexico, Chile, Spain, and at the South Pole, joined to form the Event Horizon Telescope. Each recorded radio waves from the sky, stamped with the time from an atomic clock."),
    ("eht3",
     "Petabytes of recordings were flown on hard drives to supercomputers, which combined the signals from every pair of telescopes.",
     "Petabytes of recordings were flown on hard drives to supercomputers, which combined the signals from every pair of telescopes."),

    ("four1",
     "Each pair measures one ripple of the image: a pattern of light and dark stripes, with a particular spacing and direction. Add together enough ripples, with the right strengths and offsets, and you can build any image. That is the idea behind the [Fourier](/fˈʊɹiˌA/) transform.",
     "Each pair measures one ripple of the image: a pattern of light and dark stripes, with a particular spacing and direction. Add together enough ripples, with the right strengths and offsets, and you can build any image. That is the idea behind the Fourier transform."),
    ("four2",
     "Telescopes far apart capture fine ripples. Telescopes close together capture broad ones. And as the Earth turns, every pair sweeps through new spacings and directions, tracing arcs through the space of all possible ripples.",
     "Telescopes far apart capture fine ripples. Telescopes close together capture broad ones. And as the Earth turns, every pair sweeps through new spacings and directions, tracing arcs through the space of all possible ripples."),

    ("rec1",
     "But even after a whole night, most of that space is empty. Adding up only the ripples we measured gives a dirty image, smeared by the pattern of everything we missed.",
     "But even after a whole night, most of that space is empty. Adding up only the ripples we measured gives a \"dirty image\", smeared by the pattern of everything we missed."),
    ("rec2",
     "To fill the gaps, we add what we know about real images: brightness is never negative, and the sky is mostly smooth. Then we search for the simplest image that agrees with every measurement.",
     "To fill the gaps, we add what we know about real images: brightness is never negative, and the sky is mostly smooth. Then we search for the simplest image that agrees with every measurement."),
    ("rec3",
     "And out of the gaps, a ring appears.",
     "And out of the gaps, a ring appears."),

    ("end1",
     "Four teams, working independently on the real data, all found the same thing: a ring forty-two [microarcseconds](/mˌIkɹOˈɑɹksˌɛkəndz/) across, brighter on one side, around a dark center. Just what Einstein's theory predicts.",
     "Four teams, working independently on the real data, all found the same thing: a ring 42 microarcseconds across, brighter on one side, around a dark center. Just what Einstein's theory predicts."),
    ("end2",
     "A shadow, outlined by light that skimmed the edge of no return, and escaped.",
     "A shadow, outlined by light that skimmed the edge of no return, and escaped."),
]
